def load_md(fpath):
    """Markdown 파일을 읽어오는 유틸리티 함수"""
    try:
        with open(fpath, "r", encoding="cp949") as f:
            content = f.read()
    except:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    return content

def save_txt(fpath, content):
    """텍스트 파일로 저장하는 유틸리티 함수"""
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)