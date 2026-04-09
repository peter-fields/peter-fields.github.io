---
permalink: /vocab/
title: "Vocabulary"
layout: single
author_profile: true
---

I've kept a running list of enjoyable vocab words I encounter while reading since ~2023 and try to add to it whenever I can. Some of these words are not necessarily rare or esoteric... I just like them :) AI has made it easy to organize and share, so please explore and enjoy!

---

<div id="vocab-app">

  <div style="display:flex;gap:1em;align-items:center;margin-bottom:1.2em;flex-wrap:wrap;">
    <input id="vocab-search" type="text" placeholder="Search words…"
      style="padding:0.4em 0.7em;font-size:1em;border:1px solid #ccc;border-radius:4px;width:220px;"/>
    <button id="random-btn"
      style="padding:0.4em 0.9em;font-size:0.95em;cursor:pointer;border:1px solid #aaa;border-radius:4px;background:transparent;">
      Random word
    </button>
    <span id="count-label" style="font-size:0.88em;color:#888;"></span>
  </div>

  <div id="random-card" style="display:none;margin-bottom:1.5em;border-left:3px solid #aaa;padding:0.6em 1em;background:rgba(128,128,128,0.07);border-radius:3px;">
    <span id="rand-word" style="font-size:1.15em;font-weight:bold;"></span>
    <span id="rand-pos" style="font-size:0.85em;color:#888;margin-left:0.5em;font-style:italic;"></span>
    <div id="rand-def" style="margin-top:0.3em;"></div>
    <div id="rand-ex" style="margin-top:0.25em;font-style:italic;color:#777;font-size:0.92em;"></div>
  </div>

  <div id="recently-section" style="margin-bottom:2em;">
    <h3 style="margin-bottom:0.5em;">Recently Added</h3>
    <ul id="recent-list" style="list-style:none;padding:0;margin:0;columns:2;column-gap:2em;"></ul>
  </div>

  <h3 style="margin-bottom:0.5em;">All Words <span id="list-count" style="font-size:0.8em;color:#888;font-weight:normal;"></span></h3>
  <ul id="vocab-list" style="list-style:none;padding:0;margin:0;"></ul>
  <p id="no-results" style="display:none;color:#888;font-style:italic;">No matches found.</p>

</div>

<style>
.vocab-item { border-bottom: 1px solid rgba(128,128,128,0.2); padding: 0.45em 0; cursor: pointer; }
.vocab-item:last-child { border-bottom: none; }
.vocab-item-header { display: flex; align-items: baseline; gap: 0.5em; }
.vocab-item-word { font-weight: bold; font-size: 1.05em; }
.vocab-item-pos { font-size: 0.8em; color: #888; font-style: italic; }
.vocab-item-body { display: none; padding: 0.3em 0 0.1em 0.5em; }
.vocab-item-body.open { display: block; }
.vocab-item-def { font-size: 0.95em; }
.vocab-item-ex { font-size: 0.88em; font-style: italic; color: #777; margin-top: 0.2em; }
.recent-word { font-weight: bold; }
.recent-pos { font-size: 0.8em; color: #888; font-style: italic; margin-left: 0.3em; }
</style>

<script>
(function() {
  var vocab = {{ site.data.vocab | jsonify }};

  var sorted = vocab.slice().sort(function(a, b) { return a.word.localeCompare(b.word); });

  // --- Count label ---
  document.getElementById('count-label').textContent = vocab.length + ' words';

  // --- Recently Added (newest 10) ---
  var byDate = vocab.slice().sort(function(a, b) { return b.added.localeCompare(a.added) || a.word.localeCompare(b.word); });
  var recentList = document.getElementById('recent-list');
  byDate.slice(0, 10).forEach(function(w) {
    var li = document.createElement('li');
    li.style.marginBottom = '0.2em';
    li.innerHTML = '<span class="recent-word">' + w.word + '</span><span class="recent-pos">' + w.pos + '</span>';
    recentList.appendChild(li);
  });

  // --- Full list (alphabetical) ---
  function renderList(words) {
    var ul = document.getElementById('vocab-list');
    ul.innerHTML = '';
    document.getElementById('no-results').style.display = words.length ? 'none' : 'block';
    document.getElementById('list-count').textContent = '(' + words.length + ')';
    words.forEach(function(w) {
      var li = document.createElement('li');
      li.className = 'vocab-item';
      li.innerHTML =
        '<div class="vocab-item-header">' +
          '<span class="vocab-item-word">' + w.word + '</span>' +
          '<span class="vocab-item-pos">' + w.pos + '</span>' +
        '</div>' +
        '<div class="vocab-item-body">' +
          '<div class="vocab-item-def">' + w.definition + '</div>' +
          '<div class="vocab-item-ex">\u201c' + w.example + '\u201d</div>' +
        '</div>';
      li.querySelector('.vocab-item-header').addEventListener('click', function() {
        li.querySelector('.vocab-item-body').classList.toggle('open');
      });
      ul.appendChild(li);
    });
  }
  renderList(sorted);

  // --- Search ---
  document.getElementById('vocab-search').addEventListener('input', function() {
    var q = this.value.trim().toLowerCase();
    renderList(q ? sorted.filter(function(w) { return w.word.toLowerCase().includes(q); }) : sorted);
  });

  // --- Random word ---
  document.getElementById('random-btn').addEventListener('click', function() {
    var w = vocab[Math.floor(Math.random() * vocab.length)];
    document.getElementById('rand-word').textContent = w.word;
    document.getElementById('rand-pos').textContent = w.pos;
    document.getElementById('rand-def').textContent = w.definition;
    document.getElementById('rand-ex').textContent = '\u201c' + w.example + '\u201d';
    document.getElementById('random-card').style.display = 'block';
  });
})();
</script>
