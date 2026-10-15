"""Markdown -> PDF (pandoc for markdown -> HTML, headless Chromium for HTML -> PDF).

    python tools/build_report_pdf.py            # technical_report.md  -> Tse-min_Report.pdf (max. 6 pages)
    python tools/build_report_pdf.py method     # method_description.md -> Tse-min_Method.pdf (max. 2 pages, submission form)
"""
import asyncio
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {   # name: (markdown, stylesheet, intermediate html, pdf, html title, page limit, page margins top/bottom, left/right)
    "report": ("technical_report.md", "tools/report_pdf.css", "docs/technical_report.html", "Tse-min_Report.pdf",
               "Tse-min technical report", 6, "15mm", "16mm"),
    "method": ("method_description.md", "tools/method_pdf.css", "docs/method_description.html", "Tse-min_Method.pdf",
               "Tse-min method description", 2, "12mm", "14mm"),
}
target = sys.argv[1] if len(sys.argv) > 1 else "report"
if target not in TARGETS:
    sys.exit("usage: python tools/build_report_pdf.py [%s]" % "|".join(TARGETS))
md, css, html, pdf, title, max_pages, margin_tb, margin_lr = TARGETS[target]
md, css, html, pdf = ROOT / md, ROOT / css, ROOT / html, ROOT / pdf

subprocess.run(["pandoc", str(md), "-f", "gfm", "-t", "html5", "-s", "--metadata", "title=" + title,
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
                     margin={"top": margin_tb, "bottom": margin_tb, "left": margin_lr, "right": margin_lr})
        await b.close()

asyncio.run(main())
pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
print("%s: %d pages (limit %d)" % (pdf, pages, max_pages))
if pages > max_pages:
    sys.exit("too long: %d pages > %d" % (pages, max_pages))
