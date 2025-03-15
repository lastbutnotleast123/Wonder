#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# 设置静态文件目录
app.static_folder = 'web_interface'

@app.route('/')
def index():
    return send_from_directory('web_interface', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web_interface', path)

if __name__ == '__main__':
    print("启动Web界面服务器...")
    print("请在浏览器中访问: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True) 