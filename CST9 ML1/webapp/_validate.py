"""Static validation: HTML, CSS, JS structural sanity."""
import pathlib
import re
import sys

ROOT = pathlib.Path(r"C:\xampp\htdocs\CST9 ML1")
WEBAPP = ROOT / "webapp"

html_path = WEBAPP / "templates" / "index.html"
css_path = WEBAPP / "static" / "styles.css"
js_path = WEBAPP / "static" / "app.js"

html = html_path.read_text()
css = css_path.read_text()
js = js_path.read_text()

print(f"HTML: {len(html)} bytes, {html.count(chr(10)) + 1} lines")
print(f"CSS:  {len(css)} bytes, {css.count(chr(10)) + 1} lines")
print(f"JS:   {len(js)} bytes, {js.count(chr(10)) + 1} lines")

required_ids = ["scenario-list", "live-form", "live-result", "live-scam", "live-resp",
                "live-lr", "live-rf", "live-error", "reset-btn",
                "scammer_message", "user_response"]
missing_ids = [i for i in required_ids if f'id="{i}"' not in html]
print("Missing HTML ids:", missing_ids or "none")

print("CSS open/close braces:", css.count("{"), "/", css.count("}"))

try:
    compile(js, "app.js", "exec")
    print("JS: compiles OK")
except SyntaxError as e:
    print("JS SYNTAX ERROR:", e)
    sys.exit(1)

html_classes = set()
for m in re.findall(r'class="([^"]+)"', html):
    for c in m.split():
        html_classes.add(c)
css_classes = set(re.findall(r"\.([a-zA-Z][-a-zA-Z0-9_]*)", css))
undefined = html_classes - css_classes
print("HTML classes not in CSS:", sorted(undefined) or "none")