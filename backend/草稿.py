import os
import sqlite3

file_id = 522
conn = sqlite3.connect('./database/mydata.db')
cur = conn.cursor()
filename = cur.execute("SELECT file_path FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()[0].split("/")[-1]
directory = os.getcwd() + "/articles"
print(directory, filename)
conn.close()
