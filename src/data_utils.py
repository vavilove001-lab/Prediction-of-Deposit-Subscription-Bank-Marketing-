from pathlib import Path
import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Загружает bank_fin.csv из папки data в корне проекта.
    Функция сама определяет, где находится корень (по наличию папки src).
    Никаких путей передавать не нужно.
    """
    cwd = Path.cwd()
    root = cwd

    # Поднимаемся вверх, пока не найдём папку src
    while not (root / "src").exists() and root != root.parent:
        root = root.parent

    if not (root / "src").exists():
        raise FileNotFoundError(
            "Не удалось найти корень проекта: папка 'src' не найдена ни в текущей директории, "
            "ни в родительских папках."
        )

    data_path = root / "data" / "bank_fin.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\n"
            "Проверьте, что файл 'bank_fin.csv' лежит в папке 'data' в корне проекта."
        )

    try:
        df = pd.read_csv(data_path, sep=';', encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, sep=';', encoding="latin-1")

    return df