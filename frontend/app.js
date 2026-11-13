(() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
  const markerLayer = $('markers'), seek = $('seek');
  let data = null, sample = true, sampleTime = 0, playing = false, lastTick = 0, videoUrl = null, videoFile = null, analysisToken = 0;
  const colors = { team_a: '#d7ef92', team_b: '#f2a68d', ball: '#ffffff', other: '#eacb6d' };
  const initial = [[12,34],[25,18],[27,49],[38,31],[46,53],[54,18],[63,36],[74,50],[82,22],[93,34]];
  const sampleObjects = t => initial.map((p, i) => ({ id: String(i + 1), type: 'player', team: i < 5 ? 'team_a' : 'team_b', x: Math.max(2, Math.min(98, p[0] + Math.sin(t / 3 + i) * 3)), y: Math.max(2, Math.min(98, p[1] / 68 * 100 + Math.cos(t / 4 + i) * 4)) })).concat([{type:'ball', x:50 + Math.sin(t / 3) * 16, y:50 + Math.cos(t / 3) * 17}]);
  const format = n => { n = Math.max(0, Math.floor(Number.isFinite(n) ? n : 0)); return `${String(Math.floor(n / 60)).padStart(2,'0')}:${String(n % 60).padStart(2,'0')}`; };
  const time = () => videoUrl ? video.currentTime : sampleTime;
  const duration = () => videoUrl && Number.isFinite(video.duration) ? video.duration : sample ? 20 : (data?.duration ?? data?.frames.at(-1)?.time ?? 0);
  function frameAt(t) {
    if (sample) return {objects: sampleObjects(t)};
    if (!data?.frames.length) return {objects: []};
    let lo = 0, hi = data.frames.length - 1;
    if (t < data.frames[0].time) return {objects: []};
    while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (data.frames[mid].time <= t) lo = mid; else hi = mid - 1; }
    return data.frames[lo];
  }
  function draw() {
    const t = time(), frame = frameAt(t), objects = frame.objects;
    markerLayer.replaceChildren();
    for (const o of objects) {
      if (!Number.isFinite(o.x) || !Number.isFinite(o.y) || o.x < 0 || o.x > 100 || o.y < 0 || o.y > 100) continue;
      const group = document.createElementNS('http://www.w3.org/2000/svg','g');
      const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
      circle.setAttribute('cx', 25 + o.x * 6.3); circle.setAttribute('cy', 25 + o.y * 3.9); circle.setAttribute('r', o.type === 'ball' ? 6 : 10); circle.setAttribute('fill', colors[o.type === 'ball' ? 'ball' : o.team] || colors.other); circle.setAttribute('stroke', '#244b3b'); circle.setAttribute('stroke-width', '2'); group.append(circle);
      if (o.type !== 'ball' && o.id != null) { const text = document.createElementNS('http://www.w3.org/2000/svg','text'); text.setAttribute('x',25 + o.x * 6.3); text.setAttribute('y',29 + o.y * 3.9); text.setAttribute('text-anchor','middle'); text.setAttribute('font-size','10'); text.setAttribute('font-weight','700'); text.setAttribute('fill','#284133'); text.textContent = String(o.id).slice(0,3); group.append(text); }
      markerLayer.append(group);
    }
    $('playerCount').textContent = objects.filter(o => o.type === 'player').length;
    $('ballStatus').textContent = objects.some(o => o.type === 'ball') ? 'Visible' : 'Not visible';
    $('currentTime').textContent = format(t); $('duration').textContent = format(duration()); seek.max = duration(); seek.value = Math.min(t, duration());
    $('frameStatus').textContent = sample ? 'Illustrative positions' : data ? `Frame at ${format(frame.time ?? 0)} · ${data.frames.length} frames loaded` : 'Analyze video or load projected JSON';
    drawBoxes(objects);
  }
  function drawBoxes(objects) {
    const rect = canvas.getBoundingClientRect(); canvas.width = Math.round(rect.width * devicePixelRatio); canvas.height = Math.round(rect.height * devicePixelRatio); ctx.clearRect(0,0,canvas.width,canvas.height);
    if (!videoUrl || !video.videoWidth) return;
    const scale = Math.min(canvas.width / video.videoWidth, canvas.height / video.videoHeight), w = video.videoWidth * scale, h = video.videoHeight * scale, ox = (canvas.width - w) / 2, oy = (canvas.height - h) / 2;
    objects.forEach(o => { if (!Array.isArray(o.bbox) || o.bbox.length !== 4 || !o.bbox.every(Number.isFinite)) return; const [x,y,bw,bh] = o.bbox; ctx.strokeStyle = colors[o.type === 'ball' ? 'ball' : o.team] || colors.other; ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeRect(ox+x*w,oy+y*h,bw*w,bh*h); if (o.label) { ctx.font = `${12 * devicePixelRatio}px sans-serif`; ctx.fillStyle = ctx.strokeStyle; ctx.fillText(`${o.label} ${Math.round((o.confidence || 0)*100)}%`, ox+x*w, Math.max(12 * devicePixelRatio, oy+y*h-4*devicePixelRatio)); } });
  }
  function validate(raw) {
    if (!raw || !Array.isArray(raw.frames) || !raw.frames.length) throw Error('The file needs a non-empty frames array.');
    if (raw.duration != null && (!Number.isFinite(raw.duration) || raw.duration < 0)) throw Error('Duration must be a non-negative number.');
    const frames = raw.frames.map((f,i) => { if (!Number.isFinite(f.time) || f.time < 0 || !Array.isArray(f.objects)) throw Error(`Frame ${i + 1} needs a non-negative time and an objects array.`); f.objects.forEach((o,j) => { if (!o || !['player','ball','other'].includes(o.type) || !Number.isFinite(o.x) || !Number.isFinite(o.y) || o.x < 0 || o.x > 100 || o.y < 0 || o.y > 100) throw Error(`Object ${j + 1} in frame ${i + 1} needs a valid type and x/y between 0 and 100.`); if (o.bbox != null && (!Array.isArray(o.bbox) || o.bbox.length !== 4 || !o.bbox.every(v => Number.isFinite(v) && v >= 0 && v <= 1) || o.bbox[0]+o.bbox[2]>1.000001 || o.bbox[1]+o.bbox[3]>1.000001)) throw Error(`Object ${j + 1} in frame ${i + 1} has an invalid normalized bbox.`); }); return f; }).sort((a,b) => a.time-b.time);
    return {...raw, frames};
  }
  function setNotice(message, error = false) { $('notice').textContent = message; $('notice').style.color = error ? '#a44536' : ''; $('notice').style.background = error ? '#fff0ed' : ''; }
  function hideProgress() { $('analysisProgress').hidden = true; }
  async function loadData(event) { const file = event.target.files[0]; if (!file) return; try { const next = validate(JSON.parse(await file.text())); analysisToken++; hideProgress(); $('analyzeBtn').disabled = !videoFile; $('analyzeBtn').textContent = '✳  Analyze video'; data = next; sample = false; sampleTime = 0; if (videoUrl) video.currentTime = 0; $('sessionName').textContent = next.name || file.name; $('sessionDetail').textContent = `${next.frames.length} projected frames · ${videoUrl ? 'video loaded' : 'add a video to sync footage'}`; $('dataStatus').textContent = 'Projected data'; setNotice('Projected data loaded. Positions follow the video timeline; add a video if you haven’t already.'); draw(); } catch (err) { setNotice(`Could not load tracking data: ${err.message}`, true); } event.target.value = ''; }
  function loadVideo(event) { const file = event.target.files[0]; if (!file) return; video.pause(); playing = false; analysisToken++; hideProgress(); videoFile = file; sample = false; data = null; $('dataStatus').textContent = 'Awaiting analysis'; $('analyzeBtn').disabled = false; $('analyzeBtn').textContent = '✳  Analyze video'; if (videoUrl) URL.revokeObjectURL(videoUrl); videoUrl = URL.createObjectURL(file); video.src = videoUrl; video.style.display = 'block'; $('videoEmpty').style.display = 'none'; $('videoTag').style.display = 'block'; $('videoBadge').textContent = 'VIDEO LOADED'; $('sessionDetail').textContent = `${file.name} · ready to analyze`; setNotice('Video loaded. Select Analyze video to run the local model, or load projected JSON.'); $('playBtn').textContent = '▶'; event.target.value = ''; draw(); }
  $('analyzeBtn').addEventListener('click', async () => {
    if (!videoFile) return;
    const token = ++analysisToken, file = videoFile;
    $('analyzeBtn').disabled = true; $('analyzeBtn').textContent = 'Analyzing…';
    setNotice('Analyzing your video locally. Your detections will appear on the footage when it is ready.');
    $('analysisProgress').hidden = false; $('progressTitle').textContent = 'Preparing analysis…';
    $('progressPercent').textContent = ''; $('progressFill').style.width = '0%';
    $('progressDetail').textContent = 'Uploading video to the local server';
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file });
      if (!response.ok) { const error = await response.json(); throw Error(error.error || 'Analysis failed.'); }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = '', result = null;
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
        const lines = buffer.split('\n'); buffer = lines.pop();
        if (done && buffer.trim()) lines.push(buffer);
        for (const line of lines) {
          if (!line.trim()) continue;
          const message = JSON.parse(line);
          if (token !== analysisToken) { await reader.cancel(); return; }
          if (message.event === 'error') throw Error(message.error);
          if (message.event === 'complete') result = message.result;
          if (message.event === 'progress') {
            if (message.stage === 'waiting') { $('progressTitle').textContent = 'Waiting for the model…'; $('progressDetail').textContent = 'Another analysis may still be finishing.'; }
            if (message.stage === 'loading') { $('progressTitle').textContent = 'Loading the model…'; $('progressDetail').textContent = 'This may take a moment on the first run.'; }
            if (message.stage === 'processing') {
              const {processed, total, elapsed} = message;
              const percent = total ? Math.min(99, Math.round(processed / total * 100)) : null;
              $('progressTitle').textContent = 'Analyzing video';
              $('progressPercent').textContent = percent === null ? '' : `${percent}%`;
              $('progressFill').style.width = `${percent || 0}%`;
              const remaining = total && processed >= 3 ? Math.max(0, (total - processed) * elapsed / processed) : null;
              $('progressDetail').textContent = `${processed}${total ? ` / ${total}` : ''} frames processed · ${remaining === null ? 'Estimating time remaining…' : `About ${format(remaining)} remaining`}`;
            }
          }
        }
        if (done) break;
      }
      if (!result) throw Error('The server stopped before analysis finished.');
      if (token !== analysisToken) return;
      hideProgress(); data = result; sample = false; sampleTime = 0; video.currentTime = 0;
      $('sessionName').textContent = file.name; $('sessionDetail').textContent = `${result.frames.length} analyzed frames · video loaded`;
      $('dataStatus').textContent = 'YOLO detections';
      setNotice('Analysis complete. Boxes show detections; the tactical map awaits homography.'); draw();
    } catch (err) { if (token === analysisToken) { hideProgress(); setNotice(`Analysis failed: ${err.message}. Start the dashboard using frontend/server.py.`, true); } }
    finally { if (token === analysisToken) { $('analyzeBtn').disabled = false; $('analyzeBtn').textContent = '✳  Analyze video'; } }
  });
  $('dataInput').addEventListener('change',loadData); $('videoInput').addEventListener('change',loadVideo); $('videoInputEmpty').addEventListener('change',loadVideo);
  $('sampleBtn').addEventListener('click',() => { analysisToken++; hideProgress(); $('analyzeBtn').disabled = !videoFile; $('analyzeBtn').textContent = '✳  Analyze video'; video.pause(); playing = false; sample = true; data = null; sampleTime = 0; if (videoUrl) video.currentTime = 0; $('playBtn').textContent = '▶'; $('sessionName').textContent = 'Sample session'; $('sessionDetail').textContent = videoUrl ? 'Illustrative positions · video loaded' : 'Illustrative data · no match video loaded'; $('dataStatus').textContent = 'Sample projection'; setNotice('You’re exploring sample data. Load projected JSON to view your own match.'); draw(); });
  $('playBtn').addEventListener('click',async () => { if (videoUrl) { if (video.paused) { try { await video.play(); } catch { setNotice('This video could not be played by your browser.', true); } } else video.pause(); } else { if (duration() <= 0) return; if (sampleTime >= duration()) sampleTime = 0; playing = !playing; $('playBtn').textContent = playing ? '❚❚' : '▶'; lastTick = performance.now(); } });
  video.addEventListener('play',() => $('playBtn').textContent='❚❚'); video.addEventListener('pause',() => $('playBtn').textContent='▶'); video.addEventListener('loadedmetadata',draw); video.addEventListener('timeupdate',draw); video.addEventListener('seeked',draw); video.addEventListener('error',() => setNotice('This video could not be opened. Try a browser-supported MP4 file.',true));
  seek.addEventListener('input',() => { if (videoUrl) video.currentTime = Number(seek.value); else sampleTime = Number(seek.value); draw(); }); $('muteBtn').addEventListener('click',() => { video.muted = !video.muted; $('muteBtn').textContent = video.muted ? '◖×' : '◖))'; $('muteBtn').setAttribute('aria-label',video.muted ? 'Unmute video' : 'Mute video'); });
  function tick(now) { if (videoUrl && !video.paused && !video.ended) draw(); if (playing && !videoUrl) { sampleTime = Math.min(duration(), sampleTime + (now-lastTick)/1000); if (sampleTime >= duration()) { playing = false; $('playBtn').textContent = '▶'; } draw(); } lastTick = now; requestAnimationFrame(tick); } window.addEventListener('resize',draw); draw(); requestAnimationFrame(tick);
})();
