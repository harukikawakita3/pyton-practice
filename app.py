from flask import Flask

# Flaskを初期化
app = Flask(__name__)

# ルートURLを定義
@app.route('/')
def hello_world():
    return 'Hello, world!'

# アプリケーションを実行
if __name__ == '__main__':
    app.run()