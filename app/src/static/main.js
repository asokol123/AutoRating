function set_values(sentiment, rating, error) {
  document.getElementById("sentiment").textContent = sentiment;
  document.getElementById("rating").textContent = rating;
  document.getElementById("error").textContent = error;
}

function submit() {
  var text = document.getElementById("review").value;
  var model = document.getElementById("model").value;
  var params = new URLSearchParams();
  params.set('text', text);
  params.set('model', model);

  var loader = document.getElementById("loader");
  loader.style.display = "block";
  set_values('', '', '');
  fetch(`/api/analize?${params.toString()}`, {
    method: 'GET',
    credentials: "same-origin",
  })
  .then((response) => {
    loader.style.display = "none";
    return response.json();
  })
  .then((data) => {
    if ("Error" in data) {
      set_values('', '', `Error: ${data["Error"]}`);
    } else {
      set_values(`Sentiment: ${data["sentiment"]}`, `Rating: ${data["rating"]}`, '');
    }
  })
}
