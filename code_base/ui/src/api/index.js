const isLocalHost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname)
const BASE_URL = isLocalHost
  ? 'http://localhost:8000'
  : `${window.location.protocol}//api-${window.location.host}`;
    
const routes = {
  test: '/',
  health_check: '/api/health_check',
  lesson_plan_status: '/api/lesson_planner/status',
  lesson_plan_index: '/api/lesson_planner/index_textbook',
  lesson_plan_generate: '/api/lesson_planner/generate',
  lesson_plan_pdf: '/api/lesson_planner/pdf',
  // add more routes here as needed, e.g.:
  // getStatus: '/api/video_status',
}

async function postJson(route, payload) {
  const res = await fetch(`${BASE_URL}${route}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const err = data?.detail || JSON.stringify(data) || res.statusText
    throw new Error(err)
  }
  return data
}

async function getJson(route) {
  const res = await fetch(`${BASE_URL}${route}`)
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const err = data?.detail || JSON.stringify(data) || res.statusText
    throw new Error(err)
  }
  return data
}

export { BASE_URL, routes, getJson, postJson }
