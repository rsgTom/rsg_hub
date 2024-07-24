from flask import Flask, jsonify, send_file
import os

app = Flask(__name__)

@app.route('/data/blog_posts', methods=['GET'])
def get_blog_posts():
    file_path = '../../data/raw/blog_posts_extracted.json'
    print(f"Looking for file at: {file_path}")
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
