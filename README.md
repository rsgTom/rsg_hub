# RSG Hub 🚀

RSG Hub is a comprehensive platform for extracting and displaying blog post data using a backend written in Python and a frontend built with React Native. The backend processes and serves the blog post data, while the React Native app displays the posts in an interactive and user-friendly manner.

## Project Structure 📁

rsg_hub/
├── backend/
│ ├── config/
│ │ ├── init.py
│ │ ├── config_loader.py
│ │ ├── config.yaml
│ ├── data/
│ │ ├── clean/
│ │ ├── raw/
│ │ │ ├── blog_posts.json
│ │ │ ├── blog_posts_extracted.json
│ ├── logs/
│ │ ├── app.log
│ │ ├── blog.log
│ │ ├── error.log
│ │ ├── resolute_cleaner.log
│ ├── src/
│ │ ├── init.py
│ │ ├── resolute_cleaner.py
│ │ ├── resolute_extractor.py
│ │ ├── resolute_scraper.py
│ │ ├── server.py
│ ├── .env
│ ├── .env.template
│ ├── requirements.txt
├── react-native-app/
│ ├── src/
│ │ ├── components/
│ │ │ ├── BlogPosts.js
│ │ ├── App.js
│ ├── package.json
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
├── setup.py

markdown
Copy code

## Prerequisites 📋

- Docker
- Node.js and npm
- Python 3.9 or later

## Setup 🛠️

### Backend Setup

1. **Navigate to the backend directory**:

   ```sh
   cd rsg_hub/backend
   ```

2. **Install the required Python packages**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Python extraction script**:

    ```sh
    python src/resolute_extractor.py
    ```

4. **Start the Flask server**:

    ```sh
    python src/server.py
    ```

### React Native App Setup

1. **Navigate to the rsg_hub_app directory**:

    ```sh
    cd rsg_hub/rsg_hub_app
    ```

2. **Install the required npm packages**:

    ```sh
    npm install
    ```

3. **Run the React Native app**:

    ```sh
    npx react-native run-android

    or

    npx react-native run-ios
    ```

### Docker Setup 🐳

Run the setup script to build and run the Docker containers for both the backend and the app:

```sh
./setup_docker.sh
```

### Usage 🚀

Backend: The Flask server will be running on port 5000 and can be accessed at <http://localhost:5000/data/blog_posts>.

App: The Metro Bundler will be running on port 8081. Ensure your mobile device or emulator is connected to the same network as your development machine.

### Example API Endpoint

Get Blog Posts: GET /data/blog_posts

### Environment Variables 🌍

Ensure you have a .env file in the backend directory with the necessary environment variables:

OPENAI_API_KEY=your_openai_api_key
SMART_LLM=GPT-4o
TEMPERATURE=0

## Development 🧑‍💻

Adding New Features
Backend: Add new features to the backend by modifying the scripts in the src directory. Ensure you update the Flask server endpoints as needed.

Frontend: Add new components or update existing ones in the react-native-app/src/components directory.

## Logging 📜

Logs are stored in the backend/logs directory. Ensure you monitor these logs for any errors or important information.

## Contributing 🤝

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License 📄
This project is for RSG use only

## Contact 📬

For any questions or support, please open an issue on GitHub or contact the maintainer.

Happy coding! 🎉

## Instructions for Usage

1. **Clone the repository**:

   ```sh
   git clone <your-repo-url>
   cd rsg_hub
   ```

2. **Run the setup script**:

    ```sh
    ./setup_docker.sh
    ```

3. **Access the backend**:

    Open a browser and navigate to <http://localhost:5000/data/blog_posts>.

4. **Run the App**:

    Connect your mobile device or start an emulator.

    ```sh
    cd rsg_hub_app
    npx react-native run-android
    ```

    or

    ```sh
    npx react-native run-ios
    ```

By following these instructions, you should be able to set up and run both the backend and the frontend of your project seamlessly. Let me know if you need any further adjustments or additional details!
