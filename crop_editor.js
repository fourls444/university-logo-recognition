export default function({parentElement, data, setStateValue}) {
  const root = parentElement.querySelector('.editor');
  const svg = root.querySelector('svg');
  const ns = 'http://www.w3.org/2000/svg';
  const W = data.width, H = data.height;
  // Keep local draft across component updates; new upload/open has a new key.
  let box = root.cropDraft || {...data.box};
  root.cropDraft = box;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const image = svg.querySelector('image');
  image.setAttribute('href', data.src);
  image.setAttribute('width', W); image.setAttribute('height', H);
  const selection = svg.querySelector('.selection');
  const shade = svg.querySelector('.shade');
  svg.querySelectorAll('.handle').forEach(e => e.remove());
  const handles = {};
  for (const name of ['nw','n','ne','e','se','s','sw','w']) {
    const g = document.createElementNS(ns, 'g');
    g.classList.add('handle'); g.dataset.handle = name;
    g.setAttribute('tabindex', '0'); g.setAttribute('role', 'button');
    g.setAttribute('aria-label', `Resize ${name.toUpperCase()} crop edge with arrow keys`);
    g.style.setProperty('--cursor', ['n','s'].includes(name) ? 'ns-resize' : ['e','w'].includes(name) ? 'ew-resize' : ['nw','se'].includes(name) ? 'nwse-resize' : 'nesw-resize');
    for (const cls of ['hit','dot']) { const circle = document.createElementNS(ns, 'circle'); circle.classList.add(cls); g.appendChild(circle); }
    handles[name] = g; svg.appendChild(g);
  }
  selection.setAttribute('tabindex','0'); selection.setAttribute('role','button');
  selection.setAttribute('aria-label','Move crop with arrow keys');
  const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
  const draw = () => {
    const {left:x, top:y, width:w, height:h} = box;
    for (const [attr,val] of Object.entries({x,y,width:w,height:h})) selection.setAttribute(attr,val);
    shade.setAttribute('d', `M0 0H${W}V${H}H0Z M${x} ${y}V${y+h}H${x+w}V${y}Z`);
    const scale = Math.max(.000001, svg.getScreenCTM()?.a || 1);
    const positions = {nw:[x,y],n:[x+w/2,y],ne:[x+w,y],e:[x+w,y+h/2],se:[x+w,y+h],s:[x+w/2,y+h],sw:[x,y+h],w:[x,y+h/2]};
    for (const [name,g] of Object.entries(handles)) {
      g.setAttribute('transform',`translate(${positions[name].join(' ')})`);
      g.querySelector('.hit').setAttribute('r',22/scale);
      g.querySelector('.dot').setAttribute('r',5/scale);
    }
    root.querySelector('output').textContent = `${Math.round(w)} × ${Math.round(h)} px`;
    root.querySelectorAll('input').forEach(input => {input.value = Math.round(box[input.dataset.field]);});
    root.cropDraft = {...box};
  };
  const publish = () => {root.cropDraft = {...box}; setStateValue('box', {...box});};
  const resize = (original, handle, dx, dy) => {
    let l=original.left, t=original.top, r=l+original.width, b=t+original.height;
    if (handle === 'move') {l=clamp(l+dx,0,W-original.width);t=clamp(t+dy,0,H-original.height);r=l+original.width;b=t+original.height;}
    else {if(handle.includes('w'))l=clamp(l+dx,0,r-1);if(handle.includes('e'))r=clamp(r+dx,l+1,W);if(handle.includes('n'))t=clamp(t+dy,0,b-1);if(handle.includes('s'))b=clamp(b+dy,t+1,H);}
    return {left:l,top:t,width:r-l,height:b-t};
  };
  const point = e => new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  let drag = null;
  svg.onpointerdown = e => {
    const handle = e.target.closest('[data-handle]')?.dataset.handle;
    if (!handle || e.button !== 0) return;
    e.preventDefault(); drag = {pointer:e.pointerId, start:point(e), box:{...box}, handle};
    svg.setPointerCapture(e.pointerId);
  };
  svg.onpointermove = e => {
    if (!drag || e.pointerId !== drag.pointer) return;
    const p=point(e); box=resize(drag.box,drag.handle,p.x-drag.start.x,p.y-drag.start.y); draw();
  };
  svg.onpointerup = e => {
    if (!drag || e.pointerId !== drag.pointer) return;
    drag=null; if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId); publish();
  };
  svg.onpointercancel = () => {if(drag){box=drag.box;drag=null;draw();}};
  svg.onkeydown = e => {
    const handle = e.target.closest('[data-handle]')?.dataset.handle;
    const direction = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
    if (!handle || !direction) return;
    e.preventDefault(); const step=e.shiftKey?10:1;
    box=resize(box,handle,direction[0]*step,direction[1]*step);draw();publish();
  };
  root.querySelector('.reset').onclick = () => {box={left:0,top:0,width:W,height:H};draw();publish();};
  root.querySelectorAll('input').forEach(input => {input.onchange=()=>{
    const field=input.dataset.field, value=Number(input.value);
    if(!Number.isFinite(value)){draw();return;}
    box={...box};box[field]=Math.round(value);
    box.left=clamp(box.left,0,W-1);box.top=clamp(box.top,0,H-1);
    box.width=clamp(box.width,1,W-box.left);box.height=clamp(box.height,1,H-box.top);
    draw();publish();
  };});
  const observer=new ResizeObserver(draw);observer.observe(svg);draw();
  return () => {observer.disconnect();svg.onpointerdown=null;svg.onpointermove=null;svg.onpointerup=null;svg.onpointercancel=null;svg.onkeydown=null;};
}
