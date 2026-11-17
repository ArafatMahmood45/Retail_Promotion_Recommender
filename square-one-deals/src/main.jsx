import React, {createContext,useContext,useEffect,useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {BrowserRouter,NavLink,Link,useParams} from 'react-router-dom';
import {Search,ShoppingBag,Store,Heart,UserRound,ArrowRight,CalendarDays,Tag,Sparkles,Menu,X,SlidersHorizontal,Check,Eye,ShoppingCart,RotateCcw} from 'lucide-react';
import './style.css';

const AppContext=createContext(null);
const dateStr=d=>d.toISOString().slice(0,10);
const demoDate='2026-10-20';
const promotionFile='/promotions_with_categories.csv';
const storeFile='/stores.csv';
const userFile='/users.csv';

function parseCSV(input){const rows=[];let row=[],cell='',quoted=false;for(let i=0;i<input.length;i++){const c=input[i];if(c==='"'){if(quoted&&input[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&input[i+1]==='\n')i++;row.push(cell);if(row.some(v=>v!==''))rows.push(row);row=[];cell='';}else cell+=c;}row.push(cell);if(row.some(v=>v!==''))rows.push(row);const headers=rows.shift()||[];return rows.map(r=>Object.fromEntries(headers.map((h,i)=>[h.trim(),(r[i]||'').trim()])));}
async function fetchCSV(file){const r=await fetch(file);if(!r.ok)throw new Error(`Unable to load ${file}. Copy your CSV into the public folder.`);return parseCSV(await r.text());}
const humanize=s=>String(s||'').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
function discount(p){const unit=String(p.discount_unit||'').toLowerCase();const n=Number(p.discount_value||0);const v=unit.includes('percent')||unit==='%'?`${n}%`:unit.includes('cad')||unit.includes('dollar')||unit==='$'?`$${n}`:`${n} ${p.discount_unit||''}`;return `${v} off${String(p.promotion_type||'').includes('second_item')?' second item':''}`;}
function useApp(){return useContext(AppContext);}
async function apiRequest(path,options={}){const response=await fetch(`/api${path}`,options);const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.detail||`API returned ${response.status}`);return data;}
function OfferCard({p,rank}){
  const {userId,selectedDate,refreshActivity,activity}=useApp();
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const [details,setDetails]=useState(false),[confirmPurchase,setConfirmPurchase]=useState(false);
  const records=activity.filter(e=>Number(e.promotion_id)===Number(p.promotion_id));
  const saved=records.some(e=>e.interaction_type==='save');
  const purchased=records.some(e=>e.interaction_type==='purchase');
  async function interact(type){
    if(busy)return false;
    setBusy(true);setError('');
    try{
      await apiRequest('/interactions',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({user_id:userId,promotion_id:Number(p.promotion_id),interaction_type:type,as_of:selectedDate})});
      refreshActivity();return true;
    }catch(e){setError(e.message);return false}
    finally{setBusy(false)}
  }
  async function openDetails(){setDetails(true);await interact('click')}
  async function purchase(){if(await interact('purchase')){setConfirmPurchase(false);setDetails(false)}}
  return <article className={'offer-card'+(purchased?' offer-purchased':'')}>
    <div className="offer-visual">
      <span className="offer-symbol"><Tag size={30}/></span>
      <span className="discount-pill">{discount(p)}</span>
      <button className={'save-heart'+(saved?' is-saved':'')} disabled={busy||purchased}
        onClick={()=>!saved&&interact('save')} aria-label={saved?'Saved offer':'Save offer'}
        title={saved?'Saved to your offers':'Save this offer'}>
        <Heart size={20} fill={saved?'currentColor':'none'}/>
      </button>
    </div>
    <div className="offer-body">
      {rank&&<span className="eyebrow">Recommended #{rank}</span>}
      <p className="muted tiny">{humanize(p.promotion_category)}</p>
      <h3>{p.store_name||'Store offer'}</h3>
      <p>{humanize(p.promotion_type)}</p>
      <div className="offer-foot"><span><CalendarDays size={14}/> Until {p.end_date||'—'}</span><span className="tiny">Demo offer</span></div>
      {purchased?<div className="purchase-status"><Check size={16}/> Purchased (demo)</div>:
        <div className="shop-actions">
          <button className="offer-view-btn" disabled={busy} onClick={openDetails}><Eye size={16}/> View offer</button>
          <button className="offer-buy-btn" disabled={busy} onClick={()=>setConfirmPurchase(true)}><ShoppingCart size={16}/> Purchase</button>
        </div>}
      {saved&&!purchased&&<p className="saved-caption"><Heart size={13} fill="currentColor"/> Saved to your offers</p>}
      {error&&<p className="interaction-error" role="alert">{error}</p>}
    </div>
    {details&&<div className="modal-backdrop" onClick={()=>setDetails(false)}>
      <div className="offer-modal" role="dialog" aria-modal="true" aria-label="Promotion details" onClick={e=>e.stopPropagation()}>
        <button className="modal-close" onClick={()=>setDetails(false)} aria-label="Close"><X size={20}/></button>
        <span className="eyebrow">Square One Deals · Demo promotion</span>
        <h2>{p.store_name}</h2>
        <div className="modal-discount">{discount(p)}</div>
        <p>{humanize(p.promotion_category)} · {humanize(p.promotion_type)}</p>
        <p className="muted">Available {p.start_date} to {p.end_date}</p>
        <p className="demo-note">This is a synthetic offer. No real payment or order will be processed.</p>
        <div className="modal-actions">
          <button disabled={busy||saved||purchased} onClick={()=>interact('save')}><Heart size={16} fill={saved?'currentColor':'none'}/>{saved?'Saved':'Save offer'}</button>
          <button disabled={busy||purchased} className="offer-buy-btn" onClick={()=>{setDetails(false);setConfirmPurchase(true)}}><ShoppingCart size={16}/> Simulate purchase</button>
        </div>
        {error&&<p role="alert" className="interaction-error">{error}</p>}
      </div>
    </div>}
    {confirmPurchase&&<div className="modal-backdrop" onClick={()=>setConfirmPurchase(false)}>
      <div className="offer-modal" role="dialog" aria-modal="true" aria-label="Confirm simulated purchase" onClick={e=>e.stopPropagation()}>
        <button className="modal-close" onClick={()=>setConfirmPurchase(false)} aria-label="Close"><X size={20}/></button>
        <span className="eyebrow">Demo checkout</span>
        <h2>Confirm your offer</h2>
        <p><strong>{p.store_name}</strong> · {humanize(p.promotion_category)}</p>
        <div className="modal-discount">{discount(p)}</div>
        <p className="demo-note">This records a simulated purchase for Shopper {userId}. No money is charged and no real order is placed.</p>
        <div className="modal-actions">
          <button disabled={busy} onClick={()=>setConfirmPurchase(false)}>Cancel</button>
          <button disabled={busy} className="offer-buy-btn" onClick={purchase}>{busy?'Recording...':'Confirm demo purchase'}</button>
        </div>
        {error&&<p role="alert" className="interaction-error">{error}</p>}
      </div>
    </div>}
  </article>
}
function StoreCard({s}){return <Link to={`/stores/${encodeURIComponent(s.store_id)}`} className="store-card"><div className="store-avatar"><Store size={29}/></div><div><h3>{s.store_name}</h3><p>{humanize(s.category||s.store_category)}</p></div><ArrowRight size={18}/></Link>}
function SectionHeader({kicker,title,action,href}){return <div className="section-heading"><div><span className="eyebrow">{kicker}</span><h2>{title}</h2></div>{action&&<Link to={href} className="text-link">{action} <ArrowRight size={16}/></Link>}</div>}
function Header(){const [open,setOpen]=useState(false);const {userId}=useApp();const links=[['/','Home'],['/stores','Stores'],['/for-you','For You'],['/profile','Profile']];return <><div className="announcement">Discover more. Spend smarter. <span>All offers are synthetic for demonstration.</span></div><header className="header"><div className="nav-inner"><Link className="brand" to="/"><span className="brand-icon"><ShoppingBag size={23}/></span><span>square<span className="brand-one">one</span><small>DEALS</small></span></Link><nav className={open?'nav open':'nav'}>{links.map(([path,name])=><NavLink key={path} to={path} end={path==='/'} onClick={()=>setOpen(false)}>{name}</NavLink>)}</nav><div className="nav-right"><Link to="/profile" className="shopper-chip"><UserRound size={17}/> Shopper {userId}</Link><button className="mobile-menu" aria-label="Toggle menu" onClick={()=>setOpen(!open)}>{open?<X/>:<Menu/>}</button></div></div></header></>}
function Home(){const {stores,activePromotions,loading,error}=useApp();const categories=[...new Set(stores.map(s=>s.category||s.store_category).filter(Boolean))].slice(0,6);return <><section className="hero"><div className="hero-copy"><span className="hero-label"><Sparkles size={15}/> THE MALL, ALL IN ONE PLACE</span><h1>Find your next <em>great deal.</em></h1><p>Explore stores, discover promotions, and find offers picked just for you — all in one shopping destination.</p><div className="hero-actions"><Link className="btn btn-yellow" to="/stores">Explore stores <ArrowRight size={17}/></Link><Link className="btn btn-outline" to="/for-you">For you <Sparkles size={16}/></Link></div></div><div className="hero-art" aria-hidden="true"><div className="art-circle one"/><div className="art-circle two"/><div className="art-bag"><ShoppingBag size={112} strokeWidth={1.2}/></div><div className="floating-tag">NEW DEALS <span>✦</span></div></div></section><div className="container"><div className="stats-strip"><span><Store size={20}/> {stores.length||'50'} stores to explore</span><span><Tag size={20}/> {activePromotions.length} active demo offers</span><span><Sparkles size={20}/> Personalized for you</span></div><section className="section"><SectionHeader kicker="Find your favourites" title="Shop by category" action="All stores" href="/stores"/><div className="category-grid">{categories.length?categories.map((c,i)=><Link key={c} className="category-tile" to={`/stores?category=${encodeURIComponent(c)}`}><span className="category-icon">{['✦','◈','◉','✳','◇','✿'][i%6]}</span><strong>{humanize(c)}</strong><ArrowRight size={16}/></Link>):<p className="muted">Add your store CSV to show categories.</p>}</div></section><section className="section"><SectionHeader kicker="Discover the mall" title="Featured stores" action="Browse all stores" href="/stores"/><div className="store-grid">{stores.slice(0,6).map(s=><StoreCard key={s.store_id} s={s}/>)}</div></section><section className="section"><SectionHeader kicker="Don't miss out" title="Active promotions" action="Explore stores" href="/stores"/><div className="offer-grid">{activePromotions.slice(0,4).map(p=><OfferCard key={p.promotion_id} p={p}/>)}</div>{!loading&&!activePromotions.length&&<p className="muted">No promotions found for the selected demo date.</p>}</section><section className="personal-banner"><div><span className="eyebrow">A little something for you</span><h2>Deals that match your style.</h2><p>Our recommendation model ranks promotions based on simulated shopping interactions.</p><Link className="btn btn-yellow" to="/for-you">See my recommendations <ArrowRight size={16}/></Link></div><Sparkles size={110} strokeWidth={1}/></section>{error&&<p className="notice">{error}</p>}</div></>}
function Stores(){const {stores,activePromotions}=useApp();const [q,setQ]=useState('');const [category,setCategory]=useState(new URLSearchParams(window.location.search).get('category')||'All');const categories=['All',...new Set(stores.map(s=>s.category||s.store_category).filter(Boolean))];const filtered=stores.filter(s=>(category==='All'||(s.category||s.store_category)===category)&&s.store_name?.toLowerCase().includes(q.toLowerCase()));return <main className="container page"><span className="eyebrow">Explore the mall</span><h1>Our stores</h1><p className="page-lead">Discover stores and explore their promotions, all in one place.</p><div className="filter-row"><label className="search-box"><Search size={19}/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search stores..."/></label><label className="select-box"><SlidersHorizontal size={18}/><select value={category} onChange={e=>setCategory(e.target.value)}>{categories.map(c=><option key={c} value={c}>{c==='All'?'All categories':humanize(c)}</option>)}</select></label></div><p className="result-count">{filtered.length} stores · {activePromotions.length} active demo offers</p><div className="store-grid">{filtered.map(s=><StoreCard key={s.store_id} s={s}/>)}</div>{!filtered.length&&<div className="empty">No stores found. Try another search or category.</div>}</main>}
function StoreDetail(){const {id}=useParams();const {stores,activePromotions}=useApp();const store=stores.find(s=>String(s.store_id)===decodeURIComponent(id));if(!store)return <main className="container page"><h1>Store not found</h1><Link to="/stores">Back to stores</Link></main>;const offers=activePromotions.filter(p=>String(p.store_id)===String(store.store_id));return <main className="container page"><Link to="/stores" className="back-link">← All stores</Link><div className="store-hero"><div className="store-avatar big"><Store size={44}/></div><div><span className="eyebrow">Store directory</span><h1>{store.store_name}</h1><p>{humanize(store.category||store.store_category)}</p></div></div><SectionHeader kicker="Explore offers" title={`Promotions at ${store.store_name}`}/><div className="offer-grid">{offers.map(p=><OfferCard key={p.promotion_id} p={p}/>)}</div>{!offers.length&&<div className="empty">No active promotions at this store on the selected demo date.</div>}</main>}
function ForYou(){const {userId,selectedDate,activityVersion}=useApp();const [offers,setOffers]=useState([]),[loading,setLoading]=useState(true),[error,setError]=useState('');useEffect(()=>{const ctrl=new AbortController();setLoading(true);setError('');fetch(`/api/recommendations/${userId}?top_k=10&as_of=${selectedDate}`,{signal:ctrl.signal}).then(async r=>{const d=await r.json();if(!r.ok)throw new Error(d.detail||`API returned ${r.status}`);return d}).then(d=>setOffers(d.recommendations||[])).catch(e=>{if(e.name!=='AbortError')setError(e.message)}).finally(()=>{if(!ctrl.signal.aborted)setLoading(false)});return()=>ctrl.abort()},[userId,selectedDate,activityVersion]);return <main className="container page"><span className="eyebrow"><Sparkles size={15}/> Powered by History-Aware Hybrid Neural Recommender</span><h1>Picked for you.</h1><p className="page-lead">Personalized demo promotions for Shopper {userId}, ranked using past shopping interactions.</p><div className="info-strip"><Sparkles size={20}/> Interact with offers to update your history and automatically refresh the rankings. <Link to="/profile">Switch demo profile <ArrowRight size={15}/></Link></div>{loading&&<div className="empty">Loading recommendations...</div>}{error&&<div className="empty"><strong>Could not load recommendations.</strong><p>{error}</p><p>Start FastAPI using <code>python -m uvicorn main:app --reload</code> (or your API filename).</p></div>}{!loading&&!error&&<><p className="result-count">{offers.length} recommendations · Demo date: {selectedDate}</p><div className="offer-grid">{offers.map((p,i)=><OfferCard key={p.promotion_id} p={p} rank={i+1}/>)}</div>{!offers.length&&<div className="empty">No eligible promotions for this shopper and date.</div>}</>}</main>}
function Profile(){const {userId,setUserId,users,stores,promotions,selectedDate,setSelectedDate,activityVersion,refreshActivity}=useApp();const [activity,setActivity]=useState([]),[error,setError]=useState('');const [busy,setBusy]=useState(false);useEffect(()=>{let cancelled=false;setError('');apiRequest(`/interactions/${userId}?as_of=${selectedDate}`).then(d=>{if(!cancelled)setActivity(d.interactions||[])}).catch(e=>{if(!cancelled)setError(e.message)});return()=>{cancelled=true}},[userId,selectedDate,activityVersion]);async function reset(){if(!window.confirm('Reset demo activity for this shopper on this date?'))return;setBusy(true);setError('');try{await apiRequest(`/interactions/${userId}?as_of=${selectedDate}`,{method:'DELETE'});refreshActivity()}catch(e){setError(e.message)}finally{setBusy(false)}}return <main className="container page"><span className="eyebrow">Your shopping experience</span><h1>Shopper profile</h1><p className="page-lead">Switch between synthetic shoppers to demonstrate how recommendations change.</p><div className="profile-panel"><div className="profile-head"><span className="profile-avatar"><UserRound size={32}/></span><div><h2>Shopper {userId}</h2><p>Synthetic demo account</p></div></div><label className="field"><span>Select shopper</span><select value={userId} onChange={e=>setUserId(Number(e.target.value))}>{users.map(u=><option key={u.user_id} value={u.user_id}>Shopper {u.user_id}</option>)}</select></label><label className="field"><span>Demo promotion date</span><input type="date" value={selectedDate} onChange={e=>setSelectedDate(e.target.value)}/></label><p className="muted tiny">The date controls which promotions appear. We use a historical demo date because the dataset is synthetic.</p><Link to="/for-you" className="btn btn-green">View my recommendations <ArrowRight size={17}/></Link></div><div className="activity-panel"><h2>Recorded demo activity</h2><p className="muted">{activity.length} interactions saved for Shopper {userId} on {selectedDate}. Original CSV interactions are unchanged.</p>{error&&<p className="notice">{error}</p>}{activity.length>0&&<ul className="activity-list">{activity.slice().reverse().slice(0,20).map(e=><li key={e.id}><strong>{humanize(e.interaction_type)}</strong> · {stores.find(st=>String(st.store_id)===String(promotions.find(p=>Number(p.promotion_id)===Number(e.promotion_id))?.store_id))?.store_name||'Offer'} · Promotion #{e.promotion_id}</li>)}</ul>}<button className="reset-button" disabled={busy||!activity.length} onClick={reset}>Reset this shopper's demo activity</button></div></main>}
function Footer(){return <footer className="footer"><div className="container footer-inner"><div><strong>square<span className="brand-one">one</span> deals</strong><p>A student-built retail promotion recommendation demo.</p></div><div><Link to="/">Home</Link><Link to="/stores">Stores</Link><Link to="/for-you">For You</Link><Link to="/profile">Profile</Link></div></div><div className="container disclaimer">Educational demonstration only. Synthetic promotions are not actual offers from Square One or its retailers. Not affiliated with Square One Shopping Centre.</div></footer>}
function App(){const [stores,setStores]=useState([]),[promotions,setPromotions]=useState([]),[users,setUsers]=useState([]),[userId,setUserId]=useState(1),[selectedDate,setSelectedDate]=useState(demoDate),[loading,setLoading]=useState(true),[error,setError]=useState(''),[activityVersion,setActivityVersion]=useState(0);const refreshActivity=()=>setActivityVersion(v=>v+1);useEffect(()=>{Promise.all([fetchCSV(storeFile),fetchCSV(promotionFile),fetchCSV(userFile)]).then(([s,p,u])=>{setStores(s);setPromotions(p);setUsers(u);if(u.length)setUserId(Number(u[0].user_id))}).catch(e=>setError(e.message)).finally(()=>setLoading(false))},[]);const [activity,setActivity]=useState([]);
useEffect(()=>{let active=true;setActivity([]);
  apiRequest(`/interactions/${userId}?as_of=${selectedDate}`)
    .then(d=>{if(active)setActivity(d.interactions||[])})
    .catch(()=>{if(active)setActivity([])});
  return()=>{active=false};
},[userId,selectedDate,activityVersion]);
const activePromotions=useMemo(()=>promotions.filter(p=>p.start_date?.slice(0,10)<=selectedDate&&p.end_date?.slice(0,10)>=selectedDate).map(p=>({...p,store_name:stores.find(s=>String(s.store_id)===String(p.store_id))?.store_name||'Store'})),[promotions,stores,selectedDate]);return <AppContext.Provider value={{stores,promotions,users,userId,setUserId,selectedDate,setSelectedDate,activePromotions,loading,error,activity,activityVersion,refreshActivity}}><BrowserRouter><Header/><div className="main-content"><RoutesView/></div><Footer/></BrowserRouter></AppContext.Provider>}
import {Routes,Route} from 'react-router-dom';
function RoutesView(){return <Routes><Route path="/" element={<Home/>}/><Route path="/stores" element={<Stores/>}/><Route path="/stores/:id" element={<StoreDetail/>}/><Route path="/for-you" element={<ForYou/>}/><Route path="/profile" element={<Profile/>}/><Route path="*" element={<main className="container page"><h1>Page not found</h1><Link to="/">Return home</Link></main>}/></Routes>}
createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
