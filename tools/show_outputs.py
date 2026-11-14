"""Print the text outputs of an executed notebook (for checking results without opening Jupyter)."""
import re, sys
import nbformat
sys.stdout.reconfigure(encoding="utf-8")
nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else "mother_tongue_detector.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        txt = ""
        if o.output_type == "stream":
            txt = o.text
        elif o.output_type in ("execute_result", "display_data"):
            d = o.data
            if "text/plain" in d and "text/html" in d and "<table" in d["text/html"]:
                # render HTML tables as plain rows
                html = d["text/html"]
                rows = re.findall(r"<tr.*?>(.*?)</tr>", html, re.S)
                txt = "\n".join(" | ".join(re.sub(r"<.*?>", "", cell).strip() for cell in re.findall(r"<t[hd].*?>(.*?)</t[hd]>", r, re.S)) for r in rows)
            elif "text/plain" in d:
                txt = d["text/plain"]
            elif "image/png" in d:
                txt = "[image]"
        elif o.output_type == "error":
            txt = f"ERROR {o.ename}: {o.evalue}"
        if txt.strip():
            print(f"--- cell {i}\n{txt.strip()[:3000]}")
