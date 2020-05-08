import pandas as pd
import io
from flask import make_response
import time


class GetData:
    def __init__(self,data):
        self.data = data


    def __call__(self):
        df = self.data
        out = io.StringIO()
        df.to_csv(out, index=False)
        file_name = time.strftime('%Y%m%d', time.localtime(time.time()))+ ' file.csv'
        response = make_response(out.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=%s" %file_name
        response.headers["Content-type"] = "text/csv"

        return response
