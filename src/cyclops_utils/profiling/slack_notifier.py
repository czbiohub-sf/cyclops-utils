"""Slack notifications for pipeline steps.

Posts each experiment's progress into a single Slack thread, with optional file
attachments, via the `ExperimentNotifier` context manager. Credentials come from
the environment (`SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`); set `OPS_SLACK_ENV_FILE`
to load them from a dotenv file instead. Without credentials, or without the
`cyclops_utils[slack]` extra installed, every entry point degrades to printing the
notification to the terminal.
"""

import os
import ssl
import sys
import json
from datetime import datetime

# Slack integration is an OPTIONAL extra (`cyclops_utils[slack]`): slack_sdk,
# python-dotenv, and certifi all live under that extra. This profiling module is
# imported pipeline-wide via @notify_step, so a missing optional dep must NOT
# break import -- when the extra is absent we degrade to terminal-only notices,
# exactly as we already do when the Slack tokens are unset (below).
try:
    from slack_sdk import WebClient  # the more powerful client (threading, files)
    from slack_sdk.errors import SlackApiError
    from dotenv import load_dotenv
    import certifi

    _SLACK_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    WebClient = None
    certifi = None

    class SlackApiError(Exception):
        """Stand-in so `except SlackApiError` clauses stay valid when the
        `cyclops_utils[slack]` extra (slack_sdk) is not installed."""

    def load_dotenv(*args, **kwargs):  # no-op fallback
        return False

    _SLACK_DEPS_AVAILABLE = False

# --- Thread Cache ---
# Maps experiment name -> Slack thread_ts, so messages from separate script runs
# (the orchestrator and its SLURM workers alike) land in one thread per experiment.


def _cache_file() -> str:
    """Path to the thread cache.

    Kept under the user's cache directory rather than beside this source file: a
    package directory is often read-only (wheels, containers, a shared install),
    and writing runtime state there is how this cache previously ended up tracked
    in version control. Resolved per call so the environment can be changed after
    import.

    Set OPS_SLACK_THREAD_CACHE to override the location -- point several users at
    one path to share experiment threads across accounts.
    """
    override = os.environ.get("OPS_SLACK_THREAD_CACHE")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "cyclops_utils", "slack_threads.json")


_active_notifier = None
_orchestrator_notifying = False


def get_active_notifier():
    """Returns the currently active ExperimentNotifier instance, or None."""
    return _active_notifier


def set_orchestrator_notifying(value: bool):
    """Signal that the orchestrator (PipelineRunner) is handling notifications."""
    global _orchestrator_notifying
    _orchestrator_notifying = value


def is_orchestrator_notifying() -> bool:
    """Check if the orchestrator is currently handling notifications."""
    return _orchestrator_notifying


def _load_thread_cache():
    """Loads the thread_ts cache from a JSON file."""
    path = _cache_file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # If the file is corrupted or unreadable, start fresh.
        return {}


def _save_thread_cache(cache):
    """Saves the thread_ts cache to a JSON file, creating its directory if needed."""
    path = _cache_file()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        print("Warning: Could not save Slack thread cache.")


# --- Slack client, built on first use ---
# Importing this module must not touch the network or the filesystem: it is
# reachable from decorators applied pipeline-wide, so an import that reads a
# dotenv file and calls auth_test() makes every consumer pay for Slack even when
# nothing sends a message. Credentials are read and the client is built and
# auth-checked on the first actual send instead.
_client = None
_channel_id = None
_client_ready = False


def _warn(message: str) -> None:
    """Warn on stderr, never stdout. Several Nextflow stages parse a stage's
    stdout for structured output, so a banner there corrupts the parse."""
    print(
        f"[slack_notifier] {message} Notifications will print to the terminal instead.",
        file=sys.stderr,
    )


def _load_credentials() -> None:
    """Populate the Slack env vars. Set OPS_SLACK_ENV_FILE to read them from a
    dotenv file; otherwise python-dotenv's default search applies and the ambient
    environment is used as-is."""
    env_file = os.environ.get("OPS_SLACK_ENV_FILE")
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()


def _get_target():
    """Return (client, channel_id) to send with, or (None, None) when Slack is
    unavailable or unconfigured. The client is created once, on first call."""
    global _client, _channel_id, _client_ready
    if _client_ready:
        return _client, _channel_id
    _client_ready = True

    if not _SLACK_DEPS_AVAILABLE:
        _warn("slack_sdk is not installed (install the `cyclops_utils[slack]` extra).")
        return None, None

    _load_credentials()
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        missing = [
            name
            for name, value in (("SLACK_BOT_TOKEN", token), ("SLACK_CHANNEL_ID", channel))
            if not value
        ]
        verb = "is" if len(missing) == 1 else "are"
        _warn(f"{' and '.join(missing)} {verb} not set.")
        return None, None

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        client = WebClient(token=token, ssl=ssl_context)
        client.auth_test()  # fail fast on a bad token rather than at first post
    except SlackApiError as e:
        _warn(f"could not initialize the Slack client: {e.response['error']}.")
        return None, None
    except Exception as e:
        _warn(f"could not initialize the Slack client: {type(e).__name__}: {e}.")
        return None, None

    _client, _channel_id = client, channel
    return _client, _channel_id


def notify(message: str, thread_ts: str = None):
    """
    Sends a specific message to the configured Slack channel using chat.postMessage.
    Can reply to a thread if thread_ts is provided.
    If the client isn't configured, it prints to the terminal instead.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    client, channel = _get_target()
    if client:
        try:
            # Use the chat_postMessage method, which returns the full message payload
            return client.chat_postMessage(
                channel=channel, text=full_message, thread_ts=thread_ts
            )
        except SlackApiError as e:
            print(f"Failed to send Slack message: {e.response['error']}")
            return None
        except Exception as e:
            # Transport failures (no DNS/network on compute nodes raise
            # URLError/gaierror) must never crash the pipeline.
            print(f"Failed to send Slack message: {type(e).__name__}: {e}")
            return None
    else:
        # Fallback to printing if Slack isn't configured
        print(f"--- SLACK NOTIFICATION ---\n{full_message}\n--------------------------")
        return None


def notify_file(file_path: str, comment: str = None, thread_ts: str = None, title: str = None):
    """
    Upload a file to the configured Slack channel. Replies in a thread if thread_ts is provided.
    Falls back to a terminal print when the Slack client isn't configured.
    """
    client, channel = _get_target()
    if client:
        try:
            kwargs = {
                "channel": channel,
                "file": file_path,
            }
            if comment is not None:
                kwargs["initial_comment"] = comment
            if title is not None:
                kwargs["title"] = title
            if thread_ts is not None:
                kwargs["thread_ts"] = thread_ts
            return client.files_upload_v2(**kwargs)
        except SlackApiError as e:
            print(f"Failed to upload file to Slack: {e.response['error']}")
            return None
        except Exception as e:
            print(f"Failed to upload file to Slack: {type(e).__name__}: {e}")
            return None
    else:
        print(f"--- SLACK FILE UPLOAD ---\n{file_path}\n{comment or ''}\n-------------------------")
        return None


def notify_files(files, comment: str = None, thread_ts: str = None):
    """Upload multiple files into a single Slack message via files_upload_v2's
    `file_uploads` parameter. `files` is a list where each entry is either a
    path (str/Path) or a (path, title) tuple. All files share one `comment`
    (Slack's initial_comment) and one `thread_ts`."""
    if not files:
        return None
    client, channel = _get_target()
    if client:
        try:
            file_uploads = []
            for entry in files:
                if isinstance(entry, tuple):
                    path, title = entry[0], (entry[1] if len(entry) > 1 else None)
                else:
                    path, title = entry, None
                upload = {"file": str(path)}
                if title:
                    upload["title"] = title
                file_uploads.append(upload)
            kwargs = {"channel": channel, "file_uploads": file_uploads}
            if comment is not None:
                kwargs["initial_comment"] = comment
            if thread_ts is not None:
                kwargs["thread_ts"] = thread_ts
            resp = client.files_upload_v2(**kwargs)
            try:
                ok = bool(resp.get("ok")) if resp is not None else False
            except Exception:
                ok = False
            if not ok:
                err = resp.get("error") if resp is not None else "no response"
                print(f"[notify_files] files_upload_v2 ok=False error={err!r} files={len(file_uploads)} thread_ts={thread_ts}")
            else:
                uploaded = resp.get("files") or resp.get("file") or []
                n = len(uploaded) if isinstance(uploaded, list) else 1
                print(f"[notify_files] files_upload_v2 ok=True uploaded={n} thread_ts={thread_ts}")
            return resp
        except SlackApiError as e:
            print(f"Failed to upload files to Slack: {e.response['error']}")
            return None
        except Exception as e:
            print(f"[notify_files] unexpected error: {type(e).__name__}: {e}")
            return None
    else:
        listing = "\n".join(str(f) for f in files)
        print(f"--- SLACK MULTI FILE UPLOAD ---\n{listing}\n{comment or ''}\n-------------------------")
        return None


class ExperimentNotifier:
    """A context manager to send notifications for an experiment."""

    def __init__(self, experiment_name: str, enabled: bool = True, mention_user_id: str = None):
        self.experiment_name = experiment_name
        self.enabled = enabled
        self.mention_user_id = mention_user_id or os.environ.get("SLACK_MENTION_USER_ID")
        self.thread_ts = None
        # Load the thread_ts from cache if it exists for this experiment
        if self.enabled:
            cache = _load_thread_cache()
            self.thread_ts = cache.get(self.experiment_name)

    def _send(self, message: str):
        if not self.enabled:
            return

        response = notify(message, thread_ts=self.thread_ts)

        # The WebClient's response is a dict-like object. If it's the first
        # message and it's successful, we can get the 'ts' to start a thread.
        if self.thread_ts is None and response and response.get("ok"):
            # Save the timestamp ('ts') so future messages can reply.
            self.thread_ts = response.get("ts")
            if self.thread_ts:
                cache = _load_thread_cache()
                cache[self.experiment_name] = self.thread_ts
                _save_thread_cache(cache)

    def step(self, message: str):
        """Notify that a step is in progress."""
        self._send(f"➡️ In progress: {message}...")

    def success(self, message: str):
        """Notify that a step was successful."""
        self._send(f"✅ Success: {message}.")

    def stage(self, message: str):
        """Notify that a major stage is starting, to visually group steps."""
        self._send(f"🏁 --- {message} --- 🏁")

    def error(self, message: str, exc: Exception = None):
        """Notify that a step has failed. Tags the configured user for attention.

        Only the exception type and message are posted. Full tracebacks carry
        absolute paths and usernames, which do not belong in a chat channel; the
        traceback still surfaces locally, wherever the exception is raised.
        """
        mention = f" <@{self.mention_user_id}>" if self.mention_user_id else ""
        full_message = f"🚨 ERROR: {message}{mention}"
        if exc:
            detail = f"{type(exc).__name__}: {exc}"
            if len(detail) > 300:
                detail = detail[:300] + " ..."
            full_message += f"\n```\n{detail}\n```"
        self._send(full_message)

    def finish(self):
        """Sends a final success message for the experiment. Tags the configured user."""
        mention = f" <@{self.mention_user_id}>" if self.mention_user_id else ""
        self._send(f"🎉 Experiment '{self.experiment_name}' finished successfully.{mention}")

    def attach(self, file_path: str, caption: str = None, title: str = None):
        """Upload a file (plot, log, etc.) into the experiment thread."""
        if not self.enabled:
            return
        comment = f"📎 Attached: {os.path.basename(file_path)}"
        if caption:
            comment += f" — {caption}"
        notify_file(
            file_path=file_path,
            comment=comment,
            thread_ts=self.thread_ts,
            title=title or os.path.basename(file_path),
        )

    def attach_batch(self, files, caption: str = None):
        """Upload multiple files into the experiment thread as a single Slack
        message. `files` is a list of paths or (path, title) tuples. All files
        share one initial comment."""
        if not self.enabled or not files:
            return
        normalized = []
        names = []
        for entry in files:
            if isinstance(entry, tuple):
                path, title = entry[0], (entry[1] if len(entry) > 1 else None)
            else:
                path, title = entry, None
            title = title or os.path.basename(str(path))
            normalized.append((str(path), title))
            names.append(os.path.basename(str(path)))
        comment = f"📎 Attached {len(normalized)} files"
        if caption:
            comment += f": {caption}"
        notify_files(files=normalized, comment=comment, thread_ts=self.thread_ts)

    def __enter__(self):
        global _active_notifier
        _active_notifier = self
        # Only send the "Starting" message if it's a brand new thread.
        if self.thread_ts is None:
            self._send(f"🚀 Starting experiment '{self.experiment_name}'...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_notifier
        if exc_type:
            # The custom decorator now calls notifier.error(), so this is a fallback.
            error_message = f"🚨 Uncaught ERROR in experiment '{self.experiment_name}':\n```{exc_val}```"
            self.error("An uncaught exception occurred", exc_val)
        # Success message is now sent manually via notifier.finish().
        # Return False to re-raise any exceptions.
        _active_notifier = None
        return False


def slurm_worker_run(func, experiment, args, kwargs):
    """SLURM-worker entry point that gives the worker process a live notifier.

    Without this, a function submitted via submitit runs in a fresh Python
    process where `get_active_notifier()` returns None — so any `@notify_step`
    decorator's `attachments` resolver silently no-ops. By creating an
    `ExperimentNotifier` here (which loads thread_ts from the shared
    `.slack_threads.json` cache the orchestrator wrote), attachments land in
    the experiment's existing Slack thread.

    `set_orchestrator_notifying(True)` keeps the decorator from re-posting
    step/success/error messages (the orchestrator already posts those when
    the SLURM job returns); the decorator only fires `attachments` from
    inside the worker.

    `_active_notifier` is set directly rather than via the context manager
    so we skip both the "🚀 Starting experiment" __enter__ message
    (orchestrator already sent it) and the duplicate "Uncaught ERROR"
    __exit__ message (orchestrator already reports failures from
    `job.result()`).
    """
    notifier = ExperimentNotifier(experiment, enabled=True)
    global _active_notifier
    _active_notifier = notifier
    set_orchestrator_notifying(True)
    try:
        return func(*args, **kwargs)
    finally:
        set_orchestrator_notifying(False)
        _active_notifier = None
