# Pintumex — Agente de Pintura (RAG + LangGraph) "Alura Proyecto OCI"

Asistente comercial-técnico para **Pintumex**. Consulta listas de precios y fichas técnicas en PDF, responde en lenguaje natural y calcula cuánta pintura se necesita (m² × rendimiento × manos).

Stack principal: **LangGraph** (flujo del agente), **ChromaDB** (RAG con 2 colecciones), **Cohere** (LLM + embeddings) y **Streamlit** (UI de chat).

---

## Qué resuelve

| Tipo de consulta | Qué hace el agente |
|---|---|
| **Precios** | Busca en `precios_actuales.pdf` y responde precio / presentación |
| **Fichas técnicas** | Busca en `ficha_tecnica.pdf` (rendimiento, usos, dilución, etc.) |
| **Cálculo** | Estima litros (y costo si hay precio) a partir de m², rendimiento y manos |
| **General** | Saludos o consultas fuera de dominio |

Diseñado para pedir **lo mínimo**: si el cliente pregunta precio, no le pide m² ni rendimiento.

---

## Arquitectura

```
                    ┌─────────────────┐
   Usuario ───────► │  Streamlit UI   │  (ui/app.py)
                    │  o FastAPI      │  (src/main.py)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LangGraph      │
                    │  AgentState     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌─────────┐    ┌──────────┐
         │ Router │───►│ Nodos   │───►│ Responder │
         └────────┘    │ Tool    │    └──────────┘
                       └────┬────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         precios        fichas        cálculo
         (Chroma)       (Chroma)      (tools.py)
              │             │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  chroma_db/ │  ← generado por ingest.py
              └─────────────┘
                     ▲
              data/*.pdf
```

### Flujo del grafo (LangGraph)

```
START → router → [precios | fichas | calculo | general] → responder → END
```

1. **Router** — Clasifica la intención del mensaje.
2. **Nodo precios / fichas** — Recupera chunks relevantes de Chroma y (en precios) estructura presentación → precio.
3. **Nodo cálculo** — Une ficha + precios, calcula litros y, si puede, el costo.
4. **Responder** — Genera la respuesta final en español con el contexto recuperado.

### Componentes

| Ruta | Rol |
|---|---|
| `data/precios_actuales.pdf` | Lista de precios (cambio frecuente) |
| `data/ficha_tecnica.pdf` | Fichas técnicas (cambio lento) |
| `src/rag/ingest.py` | Carga PDFs → embeddings Cohere → Chroma |
| `src/rag/retriever.py` | Búsqueda en colecciones `precios` y `fichas` |
| `src/agent/state.py` | `AgentState` (mensajes + variables de cálculo) |
| `src/agent/tools.py` | Fórmulas: litros y costo estimado |
| `src/agent/graph.py` | Grafo LangGraph (router → nodos → respuesta) |
| `src/main.py` | API FastAPI (opcional) |
| `ui/app.py` | Chat Streamlit |
| `chroma_db/` | Base vectorial persistente |

### Fórmula de cálculo

```
litros = (m² × manos / rendimiento_m²_por_litro) × (1 + merma)
```

- **Manos** por defecto: `2` si el usuario no las indica.
- **Merma** por defecto: `10%`.
- Equivalencias de presentación usadas en precios: `0.946 L ≈ 1 L`, `3.785 L ≈ 4 L`, `18 L ≈ 19 L`.

### Stack técnico

- **LLM / embeddings:** Cohere (`command-r-08-2024`, `embed-multilingual-v3.0`)
- **Orquestación:** LangGraph + LangChain
- **Vector store:** ChromaDB (2 colecciones)
- **UI:** Streamlit
- **API opcional:** FastAPI + Uvicorn

---

## Estructura del repositorio

```
pintura-agente-c/
├── data/
│   ├── precios_actuales.pdf
│   └── ficha_tecnica.pdf
├── src/
│   ├── rag/
│   │   ├── ingest.py
│   │   └── retriever.py
│   ├── agent/
│   │   ├── graph.py
│   │   ├── tools.py
│   │   └── state.py
│   └── main.py
├── ui/
│   └── app.py
├── chroma_db/                 # se genera al ingerir
├── .env.example
├── requirements.txt
└── README.md
```

---

## Cómo ejecutarlo

### 1. Requisitos

- Python 3.11+ (probado con 3.12)
- Clave de API de [Cohere](https://dashboard.cohere.com/api-keys)
- Los PDFs en `data/` (`precios_actuales.pdf`, `ficha_tecnica.pdf`)

### 2. Instalación

```bash
# Desde la raíz del proyecto
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Variables de entorno

```bash
# Windows
copy .env.example .env

# Linux / macOS
# cp .env.example .env
```

Edita `.env` y completa al menos:

```env
COHERE_API_KEY=tu_clave_aqui
COHERE_MODEL=command-r-08-2024
COHERE_EMBEDDING_MODEL=embed-multilingual-v3.0
CHROMA_PERSIST_DIR=./chroma_db
DATA_DIR=./data
```

### 4. Ingesta de documentos (obligatorio la primera vez)

Indexa los PDFs en Chroma:

```bash
python -m src.rag.ingest
```

Solo precios (cuando actualices la lista):

```bash
python -m src.rag.ingest --only precios
```

Solo fichas:

```bash
python -m src.rag.ingest --only fichas
```

> Si cambias el modelo de embeddings, borra el contenido de `chroma_db/` (excepto `.gitkeep` si existe) y vuelve a ingerir.

### 5. Interfaz de chat (recomendado)

```bash
python -m streamlit run ui/app.py
```

Abre **http://localhost:8501**.

En el sidebar verás el modelo activo y si Chroma tiene documentos indexados.

### 6. API FastAPI (opcional)

```bash
python -m src.main
# o: uvicorn src.main:app --reload --port 8000
```

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado de PDFs y Chroma |
| `POST /chat` | Chat con el agente |
| `POST /calcular` | Cálculo directo sin LLM |
| `GET /docs` | Swagger |

Ejemplo de chat:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"¿Cuál es el precio de Dry Lux SR de 1 litro?\"}"
```

---

## Ejemplos de preguntas y respuestas

Los montos exactos dependen de tu PDF de precios; aquí se ilustra el **tipo** de respuesta esperada.

### Precios

**Usuario:**  
`¿Cuáles son los precios de la pintura Dry Lux SR de 1 litro?`

**Agente (ruta: precios):**  
Responde con el precio de la presentación pedida (o equivalentes 0.946 L ≈ 1 L).  
No pide m² ni rendimiento.

---

**Usuario:**  
`Dame el precio de Esmaflex acabado brillante en 4 litros`

**Agente:**  
Indica el precio de esa presentación según la lista indexada. Si hay varias opciones cercanas, las lista y aclara.

### Fichas técnicas

**Usuario:**  
`¿Cuál es el rendimiento de Dry Lux SR?`

**Agente (ruta: fichas):**  
Cita el rendimiento (m²/L) y, si aplica, usos o recomendaciones desde la ficha técnica.

---

**Usuario:**  
`¿Cómo se diluye el primario Esmalux?`

**Agente:**  
Responde con dilución / preparación según el PDF de fichas.

### Cálculo de cantidad / costo

**Usuario:**  
`Necesito Dry Lux para 22 m², rendimiento 10 m² por litro, a dos manos. ¿Cuántos litros y cuánto cuesta?`

**Agente (ruta: calculo):**  
1. Calcula litros con la fórmula (incluye merma).  
2. Busca precio en la lista.  
3. Devuelve litros sugeridos y costo estimado.

Ejemplo numérico ilustrativo (sin merma conceptual):  
`(22 × 2 / 10) = 4.4 L` → con merma 10% ≈ `4.84 L` → redondeo sugerido (p. ej. `5.0 L`).

---

**Usuario:**  
`¿Cuánta pintura necesito para 40 m² con rendimiento 8 m²/L?`

**Agente:**  
Asume 2 manos si no se indican, calcula litros y pregunta solo lo que falte (p. ej. producto o precio) si quiere el costo.

### Qué no debería hacer

| Mal | Bien |
|---|---|
| Pedir m² cuando solo preguntan precio | Contestar precio / presentación |
| Pedir rendimiento si ya viene en el mensaje o en la ficha | Usar el dato disponible |
| Inventar precios | Decir que no está en el documento |

---

## Actualizar la lista de precios

1. Reemplaza `data/precios_actuales.pdf`.
2. Ejecuta:

```bash
python -m src.rag.ingest --only precios
```

3. Reinicia o recarga Streamlit / la API.

Las fichas solo se re-ingieren cuando cambie `ficha_tecnica.pdf`.

---

## Solución rápida de problemas

| Problema | Qué revisar |
|---|---|
| El agente “no sabe” precios/fichas | ¿Corriste `python -m src.rag.ingest`? En el sidebar: conteo de Chroma > 0 |
| Error de API / cuota Cohere | `COHERE_API_KEY` en `.env` y límites en el dashboard de Cohere |
| `streamlit` no se reconoce | Usa `python -m streamlit run ui/app.py` |
| `No module named 'src'` | Ejecuta desde la raíz del proyecto |
| Precios raros o cruzados | El PDF tabular a veces sale desordenado; valida contra la lista impresa |

---

## Screenshot app alura-proyecto en oracle OCI

# Temporalmente en http://139.177.97.203:8501/ 

<img width="1366" height="728" alt="image" src="https://github.com/user-attachments/assets/55e26198-a14b-4130-8e5a-5e90b78bca4b" />

<img width="1365" height="725" alt="image" src="https://github.com/user-attachments/assets/899fb46b-2fc6-4013-b375-0970ed23145e" />

<img width="1358" height="768" alt="image" src="https://github.com/user-attachments/assets/2c8e7331-6b1d-4edb-94c0-6071772461f3" />

<img width="1355" height="723" alt="image" src="https://github.com/user-attachments/assets/aed5384c-f34e-4112-adb1-e0ded77729a9" />

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/bf80f0b1-ac07-499a-b696-970790c2042f" />


---

## Licencia / uso

Proyecto interno de demostración para el agente comercial-técnico de Pintumex como tema en el reto de alura para el cuerso de Tech AI Builder
de oracle one next generation.  Los PDFs y precios son propiedad de quien los suministre; no se incluyen secretos en el repositorio (usa `.env` local).
