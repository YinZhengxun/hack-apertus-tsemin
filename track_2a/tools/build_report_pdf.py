"""technical_report.md -> Tse-min_Report.pdf (pandoc for markdown -> HTML, headless Chromium for HTML -> PDF).

    python tools/build_report_pdf.py
"""
import asyncio
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md, css, html, pdf = ROOT / "technical_report.md", ROOT / "tools/report_pdf.css", ROOT / "docs/technical_report.html", ROOT / "Tse-min_Report.pdf"

subprocess.run(["pandoc", str(md), "-f", "gfm", "-t", "html5", "-s", "--metadata", "title=Tse-min technical report",
                "-c", css.name, "-o", str(html)], check=True, cwd=str(ROOT))
text = html.read_text(encoding="utf-8")
text = text.replace('<link rel="stylesheet" href="%s" />' % css.name, "<style>\n%s\n</style>" % css.read_text(encoding="utf-8"))
text = text.replace('src="docs/', 'src="')            # the html lives in docs/
# tables whose cells are prose (component table) are left-aligned
text = re.sub(r'<table>(\s*<thead>\s*<tr[^>]*>\s*<th[^>]*>Component</th>)', r'<table class="left">\1', text)
# pandoc adds a title header from the metadata; drop it (the markdown has its own h1)
text = re.sub(r'<header id="title-block-header">.*?</header>', "", text, flags=re.S)
html.write_text(text, encoding="utf-8")


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(html.as_uri())
        await pg.pdf(path=str(pdf), format="A4", print_background=True, prefer_css_page_size=True,
                     display_header_footer=True, header_template="<span></span>",
                     footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                     margin={"top": "15mm", "bottom": "15mm", "left": "16mm", "right": "16mm"})
        await b.close()

asyncio.run(main())
print(pdf)
