async function askQuestion() {
  const query = document.getElementById("queryInput").value.trim();
  const responseBox = document.getElementById("responseBox");
  const loading = document.getElementById("loading");

  if (!query) {
    alert("Please enter a question");
    return;
  }

  responseBox.innerHTML = "";
  loading.innerText = "Searching multimodal RAG pipeline...";

  try {
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query: query })
    });

    const data = await response.json();

    loading.innerText = "";
    renderResponse(data);

  } catch (error) {
    loading.innerText = "";
    responseBox.innerHTML = `
      <div class="answer-card">
        <p class="error">Error connecting to backend.</p>
      </div>
    `;
    console.error(error);
  }
}


function mediaText(item) {
  if (item.description && item.description.trim() !== "") {
    return item.description;
  }

  if (item.caption && item.caption.trim() !== "") {
    return item.caption;
  }

  return "";
}


function renderResponse(data) {
  const responseBox = document.getElementById("responseBox");
  responseBox.innerHTML = "";

  if (data.message && (!data.results || data.results.length === 0)) {
    responseBox.innerHTML = `
      <div class="answer-card">
        <h3>Notice</h3>
        <p>${data.message}</p>
      </div>
    `;
    return;
  }

  if (data.type === "text") {
    responseBox.innerHTML = `
      <div class="answer-card">
        <h3>Answer</h3>
        <p>${data.answer}</p>
      </div>
    `;
  }

  else if (data.type === "image") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Images Found</h3>
          <p>No image results found for this query.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <img src="${item.url}" alt="${mediaText(item)}">
          <p>${mediaText(item)}</p>
        </div>
      `;
    });
  }

  else if (data.type === "audio") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Audio Found</h3>
          <p>No audio results found for this query.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <audio controls src="${item.url}"></audio>
          <p>${mediaText(item)}</p>
        </div>
      `;
    });
  }

  else if (data.type === "video") {
    if (!data.results || data.results.length === 0) {
      responseBox.innerHTML = `
        <div class="answer-card">
          <h3>No Videos Found</h3>
          <p>No video results found for this query.</p>
        </div>
      `;
      return;
    }

    data.results.forEach(item => {
      responseBox.innerHTML += `
        <div class="media-card">
          <video controls src="${item.url}"></video>
          <p>${mediaText(item)}</p>
        </div>
      `;
    });
  }

  else if (data.type === "mixed") {
    responseBox.innerHTML += `
      <div class="answer-card">
        <h3>Answer</h3>
        <p>${data.answer}</p>
      </div>
    `;

    if (data.message) {
      responseBox.innerHTML += `
        <div class="answer-card">
          <p>${data.message}</p>
        </div>
      `;
    }

    if (!data.media || data.media.length === 0) {
      return;
    }

    data.media.forEach(item => {
      if (item.modality === "image") {
        responseBox.innerHTML += `
          <div class="media-card">
            <img src="${item.url}" alt="${mediaText(item)}">
            <p>${mediaText(item)}</p>
          </div>
        `;
      }

      if (item.modality === "audio") {
        responseBox.innerHTML += `
          <div class="media-card">
            <audio controls src="${item.url}"></audio>
            <p>${mediaText(item)}</p>
          </div>
        `;
      }

      if (item.modality === "video") {
        responseBox.innerHTML += `
          <div class="media-card">
            <video controls src="${item.url}"></video>
            <p>${mediaText(item)}</p>
          </div>
        `;
      }
    });
  }

  else {
    responseBox.innerHTML = `
      <div class="answer-card">
        <p>Unknown response type.</p>
      </div>
    `;
  }
}