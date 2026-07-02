import os
import zipfile
import xml.etree.ElementTree as ET

def docx_to_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # The XML namespaces
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = []
                for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if t.text:
                        texts.append(t.text)
                if texts:
                    paragraphs.append(''.join(texts))
                else:
                    # Keep empty paragraphs as spacing
                    paragraphs.append('')
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

def convert_all():
    data_dir = "e:/AI_Resume_Ranker/data"
    for file in os.listdir(data_dir):
        if file.endswith(".docx"):
            docx_path = os.path.join(data_dir, file)
            txt = docx_to_text(docx_path)
            md_name = file.replace(".docx", ".md")
            md_path = os.path.join(data_dir, md_name)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"Converted {file} -> {md_name}")

if __name__ == "__main__":
    convert_all()
