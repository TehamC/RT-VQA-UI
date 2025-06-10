# RT-VQA-UI: Real-Time Video Question Answering User Interface

This repository contains the user interface (UI) for a Real-Time Video Question Answering (RT-VQA) application. This web app allows users to interact with video content by posing questions and receiving answers in real-time. It's designed as a complementary interactive inference tool for my previous VQA project: https://github.com/TehamC/VQA_Construction. By adjusting the backend, you can easily integrate other custom VQA models.

## Features

* **Video Playback:** Seamlessly play local video files.
* **Prompt Input:** Enter questions or prompts at any point during video playback.
* **Real-Time Interaction:** Get answers to your prompts while the video is playing or paused.

## Installation

To get this application up and running on your local machine, follow these steps:

1.  **Clone the Repository:**
    First, clone the project from GitHub to your local machine using either HTTPS or SSH:

    ```bash
    # Using HTTPS
    git clone [https://github.com/TehamC/RT-VQA-UI.git](https://github.com/TehamC/RT-VQA-UI.git)

    # Or, using SSH (recommended)
    git clone git@github.com:TehamC/RT-VQA-UI.git
    ```

2.  **Navigate into the Project Directory:**

    ```bash
    cd RT-VQA-UI
    ```

3.  **Install Frontend Dependencies:**
    The frontend is built with React. Navigate into the `frontend` directory and install its dependencies:

    ```bash
    cd frontend
    npm install
    cd .. # Go back to the root directory
    ```

4.  **Install Backend Dependencies:**
    The backend is built with Python (using FastAPI/Uvicorn). Navigate into the `backend` directory and install its dependencies. It's highly recommended to use a virtual environment.

    ```bash
    cd backend
    # Install Python dependencies
    pip install -r requirements.txt # coming soon
    cd .. # Go back to the root directory
    ```

## Running the Application

To run the full application, you need to start both the backend and the frontend servers concurrently.

1.  **Start the Backend Server:**
    Open your first terminal window. Navigate to the `backend` directory, activate your virtual environment, and then run the Uvicorn server:

    ```bash
    cd backend
    uvicorn main:app --reload
    ```
    The backend server will typically run on `http://127.0.0.1:8000` (or `localhost:8000`).

2.  **Start the Frontend Development Server:**
    Open a second terminal window. Navigate to the `frontend` directory and start the React development server:

    ```bash
    cd frontend
    npm run dev
    ```
    The frontend application will typically open in your browser at `http://localhost:5173` (or a similar port).

---

## Usage

Here's how to use the RT-VQA UI (Consider that video fps is low due to long VQA inference times, for each frame >1s):

### 1. Select Video File

Begin by selecting a video file from your local machine to load into the application.

![Select Video File](ui_gif/ui1.gif)

### 2. Pause Video and Enter Prompts

You can pause the video at any point to enter your questions or prompts. The system will process your input based on the current video frame. When paused, the LLM will respond with its prediction via chat. To prevent overwhelming the chat, this LLM prediction feature is disabled when the video is actively playing, as a message would appear for each new frame.

![Pause Video and Enter Prompts - Example 1](ui_gif/ui2.gif)

![Pause Video and Enter Prompts - Example 2](ui_gif/ui3.gif)

### 3. Use Prompt While Video is Playing

For real-time interaction, you can also enter prompts while the video is actively playing. The application will attempt to answer questions relevant to the ongoing video segment.

![Use Prompt While Video is Playing](ui_gif/ui4.gif)

---

## To-Do

* Fix YouTube URL support.
* add requirements
* include LLM and Yolo models 
