import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import { BASE_URL, getJson, postJson, routes } from './api'

const SUBJECT = 'CBC Mathematics'
const GRADE = 'Grade 10'

const FORMULA_TOKEN = /(\$\$[^$]+\$\$|\$[^$]+\$)/g

const normalizeLineBreaks = (value) =>
  String(value || '')
    .replace(/\r\n/g, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\*\*/g, '')
    .replace(/^#{1,6}\s*/gm, '')
    .trim()

const normalizeMcqOption = (value) => {
  let option = String(value || '').trim().replace(/^\s*[-*]\s*/, '')
  const labelPattern = /^\s*(?:Option\s*)?(?:[A-Da-d]|[1-4])[).:-]\s*/i
  while (labelPattern.test(option)) {
    option = option.replace(labelPattern, '').trim()
  }
  return option
}

const toDisplayLines = (value) => {
  let text = normalizeLineBreaks(value)
  if (!text) return []

  text = text.replace(/\s+(Step\s*\d+\s*:)/gi, '\n$1')
  text = text.replace(/\s+(Concept:|Example:|Try:|Practice:|Check:|Formula:|Why it works:|Think:)/gi, '\n$1')
  text = text.replace(/\s+(\d+\.)\s+/g, '\n$1 ')
  text = text.replace(/^\s*[-*]\s+/gm, '')

  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
}

const lineClassName = (line) => {
  if (/^Step\s*\d+\s*:/i.test(line)) return 'flow-line flow-step-title'
  if (/^(Concept|Example|Try|Practice|Check|Formula|Why it works|Think):/i.test(line)) return 'flow-line flow-label-line'
  return 'flow-line'
}

const renderFormulaAwareLine = (line) => {
  const parts = line.split(FORMULA_TOKEN)
  return parts.map((part, index) => {
    const isFormula = /^(\$\$[^$]+\$\$|\$[^$]+\$)$/.test(part)
    if (isFormula) {
      const formulaText = part.replace(/^\$\$?|\$\$?$/g, '')
      return (
        <span key={`${part}-${index}`} className="formula-chip">
          {formulaText}
        </span>
      )
    }
    return <span key={`${part}-${index}`}>{part}</span>
  })
}

function App() {
  const [topic, setTopic] = useState('')
  const [description, setDescription] = useState('')
  const [objectivesText, setObjectivesText] = useState('')
  const [durationMinutes, setDurationMinutes] = useState(40)

  const [plan, setPlan] = useState(null)
  const [sources, setSources] = useState([])
  const [ragStatus, setRagStatus] = useState({ indexed: false, record_count: 0 })

  const [loadingStatus, setLoadingStatus] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const presentationNotes = plan?.presentation_notes || plan?.notes_for_teacher || ''

  const objectives = useMemo(
    () =>
      objectivesText
        .split('\n')
        .map((item) => item.trim())
        .filter(Boolean),
    [objectivesText]
  )

  const loadRagStatus = useCallback(async () => {
    try {
      setLoadingStatus(true)
      const data = await getJson(routes.lesson_plan_status)
      setRagStatus({
        indexed: Boolean(data.indexed),
        record_count: Number(data.record_count || 0),
      })
    } catch (err) {
      setError(err.message || 'Failed to load RAG status.')
    } finally {
      setLoadingStatus(false)
    }
  }, [])

  useEffect(() => {
    loadRagStatus()
  }, [loadRagStatus])

  const handleIndex = async (forceReindex = false) => {
    setError('')
    setStatus('')
    try {
      setIndexing(true)
      const data = await postJson(routes.lesson_plan_index, { force_reindex: forceReindex })
      setStatus(data.message || 'Textbook index updated.')
      await loadRagStatus()
    } catch (err) {
      setError(err.message || 'Failed to index textbook.')
    } finally {
      setIndexing(false)
    }
  }

  const handleGenerate = async () => {
    setError('')
    setStatus('')

    if (!topic.trim()) {
      setError('Please enter a topic.')
      return
    }

    const payload = {
      subject: SUBJECT,
      grade: GRADE,
      topic: topic.trim(),
      description: description.trim(),
      objectives,
      duration_minutes: Number(durationMinutes) || 40,
    }

    try {
      setGenerating(true)
      const data = await postJson(routes.lesson_plan_generate, payload)
      setPlan(data.plan || null)
      setSources(data.sources || [])
      setStatus('Lesson plan generated.')
    } catch (err) {
      setError(err.message || 'Failed to generate lesson plan.')
    } finally {
      setGenerating(false)
    }
  }

  const handleDownloadPdf = async () => {
    if (!plan) return

    setError('')
    setStatus('')

    try {
      setDownloading(true)
      const res = await fetch(`${BASE_URL}${routes.lesson_plan_pdf}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => null)
        throw new Error(data?.detail || res.statusText)
      }
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${(plan.topic || 'lesson-plan').replace(/\s+/g, '-')}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
      setStatus('PDF downloaded.')
    } catch (err) {
      setError(err.message || 'Failed to download PDF.')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">CBC Grade 10</p>
          <h1>Maths Lesson Planner</h1>
          <p className="subhead">
            Generates presentation-ready lesson flow from textbook RAG context with cleaner classroom language.
          </p>
        </div>
        <div className="pill">
          {loadingStatus ? 'Checking index...' : ragStatus.indexed ? `Indexed: ${ragStatus.record_count} chunks` : 'Index not ready'}
        </div>
      </header>

      <div className="layout">
        <section className="card form-card">
          <h2>Plan Inputs</h2>

          <div className="index-actions">
            <button className="secondary" onClick={() => handleIndex(false)} disabled={indexing || generating || downloading}>
              {indexing ? 'Indexing...' : 'Index Textbook'}
            </button>
            <button className="ghost" onClick={() => handleIndex(true)} disabled={indexing || generating || downloading}>
              Re-index
            </button>
          </div>
          <p className="hint">Index once before generating plans. Re-index after textbook updates.</p>

          <label>
            Topic
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Congruence tests for triangles"
            />
          </label>

          <label>
            Detailed topic description
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe class context and what should be explained on screen."
            />
          </label>

          <label>
            Objectives (one per line)
            <textarea
              value={objectivesText}
              onChange={(e) => setObjectivesText(e.target.value)}
              placeholder="Identify SSS, SAS, ASA congruence tests&#10;Apply each test to solve problems"
            />
          </label>

          <label>
            Class length (minutes)
            <input
              type="number"
              value={durationMinutes}
              onChange={(e) => setDurationMinutes(e.target.value)}
              min={20}
              max={180}
            />
          </label>

          <button className="primary" onClick={handleGenerate} disabled={generating || indexing || downloading}>
            {generating ? 'Generating...' : 'Generate Lesson Plan'}
          </button>

          {error && <p className="status error">{error}</p>}
          {status && <p className="status success">{status}</p>}
        </section>

        <section className="card preview-card">
          <div className="preview-header">
            <div>
              <h2>Lesson Plan</h2>
              <p className="hint">Formatted for screen presentation</p>
            </div>
            <button className="secondary" onClick={handleDownloadPdf} disabled={!plan || generating || downloading}>
              {downloading ? 'Preparing PDF...' : 'Download PDF'}
            </button>
          </div>

          {!plan && (
            <div className="empty-state">
              <p>Generate a lesson plan to preview it here.</p>
            </div>
          )}

          {plan && (
            <div className="plan-preview">
              <div className="section-block">
                <h3>{plan.title}</h3>
                <p className="hint">
                  {plan.subject} • {plan.grade}
                </p>
                <div className="text-block">
                  {toDisplayLines(plan.overview).map((line, idx) => (
                    <p key={`overview-${idx}`} className="flow-line">
                      {renderFormulaAwareLine(line)}
                    </p>
                  ))}
                </div>
              </div>

              <div className="section-block">
                <h3>Objectives</h3>
                <ul className="plain-list">
                  {(plan.objectives || []).map((item, idx) => (
                    <li key={`${item}-${idx}`}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="section-block">
                <h3>Lesson Flow</h3>
                {(plan.lesson_flow || []).map((step, idx) => (
                  <div key={`${step.title}-${idx}`} className="flow-card">
                    <div className="flow-head">
                      <strong>{step.title}</strong>
                    </div>
                    <div className="flow-content">
                      {toDisplayLines(step.content).map((line, lineIdx) => (
                        <p key={`${step.title}-${lineIdx}`} className={lineClassName(line)}>
                          {renderFormulaAwareLine(line)}
                        </p>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {(plan.mcq_exercises || []).length > 0 && (
                <div className="section-block">
                  <h3>MCQ Exercises</h3>
                  {(plan.mcq_exercises || []).map((mcq, idx) => (
                    <div key={`${mcq.question}-${idx}`} className="mcq-preview">
                      <p className="mcq-question">
                        {idx + 1}. {mcq.question}
                      </p>
                      <ul className="plain-list">
                        {(mcq.options || []).map((opt, optIdx) => (
                          <li key={`${opt}-${optIdx}`}>
                            {String.fromCharCode(65 + optIdx)}. {normalizeMcqOption(opt)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}

              {(plan.resources || []).length > 0 && (
                <div className="section-block">
                  <h3>Resources</h3>
                  <ul className="plain-list">
                    {(plan.resources || []).map((item, idx) => (
                      <li key={`${item}-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {presentationNotes && (
                <div className="section-block">
                  <h3>Quick Recap</h3>
                  <div className="text-block">
                    {toDisplayLines(presentationNotes).map((line, idx) => (
                      <p key={`notes-${idx}`} className="flow-line">
                        {renderFormulaAwareLine(line)}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              {sources.length > 0 && (
                <div className="sources">
                  <h3>Textbook Context Used</h3>
                  {sources.slice(0, 8).map((src, idx) => (
                    <div key={src.id || idx} className="source-item">
                      <p className="source-title">
                        {src.metadata?.chapter || 'Chapter'} {src.metadata?.section ? `• ${src.metadata.section}` : ''}
                      </p>
                      <p className="source-body">{src.content}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
