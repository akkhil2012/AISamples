"""
PII Guardrail — Android app entry point (Kivy).

The guardrail engine runs 100% on-device (pure Python, no network needed).
Optionally, sanitized text can be forwarded to a server running the ADK
agent (see pii_guardrails/adk_agent.py) — raw PII never leaves the phone.

Build to APK with buildozer (see README.md).
"""

import json
import threading

from kivy.app import App
from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.utils import platform

from pii_guardrails.guardrail_engine import PIIGuardrail, Action

# Optional: forward sanitized text to your ADK agent server (adk api_server)
ADK_SERVER_URL = ""  # e.g. "http://10.0.2.2:8000/run"  — leave empty for offline mode

SEVERITY_COLORS = {
    "CRITICAL": (0.90, 0.22, 0.21, 1),
    "HIGH": (0.96, 0.49, 0.00, 1),
    "MEDIUM": (0.98, 0.75, 0.18, 1),
    "LOW": (0.55, 0.76, 0.29, 1),
}

SAMPLE = (
    "Hi, I'm Ravi. Reach me at ravi.k@example.com or +91 98765 43210. "
    "My PAN is ABCDE1234F and my card number is 4111 1111 1111 1111."
)


class GuardrailUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=12, spacing=10, **kwargs)

        self.guard = PIIGuardrail()

        self.add_widget(
            Label(
                text="[b]PII Guardrail[/b]  (on-device)",
                markup=True,
                size_hint_y=None,
                height=44,
                font_size="20sp",
            )
        )

        # Redaction style picker
        row = BoxLayout(size_hint_y=None, height=44, spacing=8)
        row.add_widget(Label(text="Redaction:", size_hint_x=0.35))
        self.style_spinner = Spinner(
            text="tag", values=("tag", "mask", "hash"), size_hint_x=0.65
        )
        row.add_widget(self.style_spinner)
        self.add_widget(row)

        # Input box
        self.input_box = TextInput(
            text=SAMPLE,
            hint_text="Paste or type text to scan for PII…",
            size_hint_y=0.32,
            font_size="15sp",
        )
        self.add_widget(self.input_box)

        # Buttons
        btn_row = BoxLayout(size_hint_y=None, height=52, spacing=8)
        scan_btn = Button(text="Scan & Sanitize", background_color=(0.13, 0.45, 0.85, 1))
        scan_btn.bind(on_release=self.on_scan)
        clear_btn = Button(text="Clear", size_hint_x=0.35)
        clear_btn.bind(on_release=lambda *_: setattr(self.input_box, "text", ""))
        btn_row.add_widget(scan_btn)
        btn_row.add_widget(clear_btn)
        self.add_widget(btn_row)

        # Results area
        scroll = ScrollView()
        self.results = Label(
            text="Results will appear here.",
            markup=True,
            size_hint_y=None,
            halign="left",
            valign="top",
            font_size="14sp",
            padding=(8, 8),
        )
        self.results.bind(
            width=lambda inst, w: setattr(inst, "text_size", (w, None)),
            texture_size=lambda inst, ts: setattr(inst, "height", ts[1]),
        )
        scroll.add_widget(self.results)
        self.add_widget(scroll)

    # ------------------------------------------------------------------

    def on_scan(self, *_):
        text = self.input_box.text.strip()
        if not text:
            self.results.text = "[i]Nothing to scan.[/i]"
            return

        self.guard.redaction_style = self.style_spinner.text
        result = self.guard.apply(text)
        self.render_result(result)

        # Offline-first: only sanitized text ever leaves the device
        if ADK_SERVER_URL and not result.blocked:
            threading.Thread(
                target=self._forward_to_agent, args=(result.sanitized_text,), daemon=True
            ).start()

    def render_result(self, result):
        lines = []
        if result.blocked:
            lines.append("[b][color=e53935]⛔ BLOCKED[/color][/b] — critical PII found.\n")
        elif result.findings:
            lines.append("[b][color=fb8c00]⚠ REDACTED[/color][/b]\n")
        else:
            lines.append("[b][color=43a047]✔ CLEAN[/color][/b] — no PII detected.\n")

        for f in result.findings:
            r, g, b, _ = SEVERITY_COLORS.get(f.severity, (1, 1, 1, 1))
            hexc = f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            lines.append(
                f"[color={hexc}]■[/color] [b]{f.entity}[/b]  "
                f"({f.severity})  {f.masked()}  →  {f.action.value}"
            )

        if not result.blocked and result.findings:
            lines.append("\n[b]Sanitized output:[/b]")
            lines.append(result.sanitized_text)

        self.results.text = "\n".join(lines)

    # ------------------------------------------------------------------

    def _forward_to_agent(self, sanitized_text: str):
        """POST sanitized text to an ADK api_server (optional)."""
        try:
            import urllib.request

            payload = json.dumps(
                {
                    "app_name": "pii_guardrails",
                    "user_id": "android_user",
                    "session_id": "s1",
                    "new_message": {
                        "role": "user",
                        "parts": [{"text": sanitized_text}],
                    },
                }
            ).encode()
            req = urllib.request.Request(
                ADK_SERVER_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                answer = resp.read().decode()
            self._append_agent_reply(answer[:2000])
        except Exception as exc:  # noqa: BLE001
            self._append_agent_reply(f"[agent unreachable: {exc}]")

    @mainthread
    def _append_agent_reply(self, text: str):
        self.results.text += f"\n\n[b]Agent reply:[/b]\n{text}"


class PIIGuardrailApp(App):
    title = "PII Guardrail"

    def build(self):
        if platform not in ("android", "ios"):
            Window.size = (420, 760)
        return GuardrailUI()


if __name__ == "__main__":
    PIIGuardrailApp().run()
