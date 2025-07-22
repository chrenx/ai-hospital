import os
import glob
import html
import markdown

# root_folder = "res/ai-discharge/2025_07_07_04_17_57_30_discharge_DeepSeek-V3/medical-cases"  # Change this to your directory
# output_html = "res/ai-discharge/2025_07_07_04_17_57_30_discharge_DeepSeek-V3/all_cases.html"

def read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    
def generate_html(root_folder, output_html, basename):

    sections = []
    case_folders = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])

    for case_id in case_folders:
        case_path = os.path.join(root_folder, case_id)
        report_md = os.path.join(case_path, f"{case_id}_user_friendly_report_en.md")
        record_en = os.path.join(case_path, f"{case_id}_orig_medical_record_en.txt")
        record_zh = os.path.join(case_path, f"{case_id}_orig_medical_record_zh.txt")
        
        report_content_raw = read_file(report_md)
        if report_content_raw is not None:
            report_content = markdown.markdown(report_content_raw)
        else:
            report_content = "<i>Missing or unreadable markdown</i>"

        record_en_content_raw = read_file(record_en)
        record_en_content = (
            f"<pre>{html.escape(record_en_content_raw)}</pre>" if record_en_content_raw is not None else "<i>Missing or unreadable EN text</i>"
        )
        record_zh_content_raw = read_file(record_zh)
        record_zh_content = (
            f"<pre>{html.escape(record_zh_content_raw)}</pre>" if record_zh_content_raw is not None else "<i>Missing or unreadable ZH text</i>"
        )

        section_html = f"""
        <section style="margin-bottom: 40px;">
        <h2>Case ID: {case_id}</h2>
        <div style="display: flex; gap: 16px;">
            <div style="flex: 1; border: 1px solid #ccc; padding: 8px;">
            <h3>User Friendly Report (Markdown)</h3>
            {report_content}
            </div>
            <div style="flex: 1; border: 1px solid #ccc; padding: 8px;">
            <h3>Original Medical Record (EN)</h3>
            {record_en_content}
            </div>
            <div style="flex: 1; border: 1px solid #ccc; padding: 8px;">
            <h3>Original Medical Record (ZH)</h3>
            {record_zh_content}
            </div>
        </div>
        </section>
        """
        sections.append(section_html)

    full_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>All Cases Horizontally</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f8f8f8;
                margin: 0;
                padding: 0 2vw;
            }}
            h2 {{
                margin-top: 32px;
                margin-bottom: 8px;
                color: #245;
                border-bottom: 1px solid #ccd;
                padding-bottom: 4px;
            }}
            h3 {{
                margin-top: 0;
                font-size: 1.1em;
                color: #444;
                border-bottom: 1px solid #eee;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-all;
                font-family: monospace;
                font-size: 14px;
                margin: 0;
            }}
            section {{
                background: #fff;
                border-radius: 7px;
                box-shadow: 0 2px 6px #0001;
                padding: 12px 8px 16px 8px;
            }}
        </style>
    </head>
    <body>
        <h1>All Cases: {basename}</h1>
        {''.join(sections)}
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Done. Output written to {output_html}")