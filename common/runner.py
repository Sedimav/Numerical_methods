import argparse
import contextlib
import io
from pathlib import Path

from common.helpers import read_json


def run_program(run, description, filename):
    base = Path(filename).resolve().parent
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input', nargs='?', type=Path, default=base/'data.json')
    parser.add_argument('--output', type=Path, default=base/'results')
    args = parser.parse_args()
    try:
        data = read_json(args.input)
        args.output.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run(data,args.output)
        report = buffer.getvalue()
        (args.output/'report.txt').write_text(report,encoding='utf-8')
        print(report,end='')
    except (ValueError,KeyError,TypeError,SyntaxError,OSError,ZeroDivisionError,OverflowError) as error:
        parser.exit(1,'Ошибка: '+str(error)+'\n')
