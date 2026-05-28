/*
script.js

This file handles frontend logic for the Multimodal RAG Pipeline.

Main responsibilities:
- Read user query from input box
- Send query to FastAPI backend
- Receive backend response
- Render text, image, audio, or video output
- Show loading and error messages

Frontend display rules:
- text response shows answer and related media
- image response shows only images
- audio response shows only audio
- video response shows only video
*/

async function askQuestion() {
  // Read query from input box
  const query = document.getElementById("queryInput").value.trim();

  // Get response and loading containers
  const responseBox = document.getElementById("responseBox");
  const loading = document.getElementById("loading");

  // Do not allow empty query
  if (!query) {
    alert("Please enter a question");
    return;
  }

  // Clear previous response
  responseBox.innerHTML = "";

  // Show loading message
  loading.innerText = "Searching multimodal RAG pipeline...";

  try {
    // Send query to FastAPI backend
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    });

    // Convert backend response to JSON
    const data = await response.json();

    // Remove loading message
    loading.innerText = "";

    // Render response based on type
    renderResponse(data);

  } catch (error) {
    // Show backend connection error
    loading.innerText = "";
    responseBox.innerHTML = `
      <div class="answer-card">
        <p class="error">Error connecting to backend.</p>
      </div>
    `;

    console.error(error);
  }
}


function renderResponse(data) {
  // Get response container
  const responseBox = document.getElementById("responseBox");

  // Clear old response
  responseBox.innerHTML = "";

  // Show backend message if no results are available
  if (data.message && (!data.results || data.results.length === 0)) {
    responseBox.innerHTML = `
      <div class="answer-card">
        <h3>Notice</h3>
        <p>${data.message}</p>
      </div>
    `;
    return;
  }

  // Render text response
  if (data.type === "text") {
    responseBox.innerHTML = `
      <div class="answer-card">
        <h3>Answer</h3>
        <p>${data.answer || ""}</p>
      </div>
    `;

    // Render related media for text query if available
    if (data.media && data.media.length > 0) {
      data.media.forEach(item => {

        // Render image
        if (item.modality === "image") {
          responseBox.innerHTML += `
            <div class="media-card">
              <img src="${item.url}" alt="">
            </div>
          `;
        }

        // Render audio
        if (item.modality === "audio") {
          responseBox.innerHTML += `
            <div class="media-card">
              <audio controls src="${item.url}"></audio>
            </div>
          `;
        }

        // Render video
        if (item.modality === "video") {
          responseBox.innerHTML += `
            <div class="media-card">
              <video controls preload="metadata" width="700">
                <source src="${item.url}">
                Your browser does not support video.
              </video>
            </div>
          `;
        }
      });
    }

    return;
  }

  // Render image-only response
  if (data.type === "image") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Images Found</h3>
          <p>No image results found.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <img src="${item.url}" alt="">
        </div>
      `;
    });

    return;
  }

  // Render audio-only response
  if (data.type === "audio") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Audio Found</h3>
          <p>No audio results found.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <audio controls src="${item.url}"></audio>
        </div>
      `;
    });

    return;
  }

  // Render video-only response
  if (data.type === "video") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Videos Found</h3>
          <p>No video results found.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <video controls preload="metadata" width="700">
            <source src="${item.url}">
            Your browser does not support video.
          </video>
        </div>
      `;
    });

    return;
  }

  // Render mixed response if mixed type is returned
  if (data.type === "mixed") {
    if (!data.media || data.media.length === 0) {
      return;
    }

    data.media.forEach(item => {

      // Render image
      if (item.modality === "image") {
        responseBox.innerHTML += `
          <div class="media-card">
            <img src="${item.url}" alt="">
          </div>
        `;
      }

      // Render audio
      if (item.modality === "audio") {
        responseBox.innerHTML += `
          <div class="media-card">
            <audio controls src="${item.url}"></audio>
          </div>
        `;
      }

      // Render video
      if (item.modality === "video") {
        responseBox.innerHTML += `
          <div class="media-card">
            <video controls preload="metadata" width="700">
              <source src="${item.url}">
              Your browser does not support video.
            </video>
          </div>
        `;
      }
    });

    return;
  }

  // Fallback for unknown response type
  responseBox.innerHTML = `
    <div class="answer-card">
      <p>Unknown response type.</p>
    </div>
  `;
}