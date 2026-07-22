[app]
title = PII Guardrail
package.name = piiguardrail
package.domain = com.akkhil
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 1.0.0

# Guardrail engine is pure stdlib -> only kivy needed on-device.
# (google-adk stays server-side; see README "Architecture" section.)
requirements = python3,kivy==2.3.0

orientation = portrait
fullscreen = 0

# Internet only needed if you enable ADK_SERVER_URL forwarding in main.py.
android.permissions = INTERNET

android.api = 34
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = False

# Kivy needs this on modern devices (incl. 16KB page-size handsets)
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
