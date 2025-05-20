from pathlib import Path

import llm4mdg

NANOID_LENGTH = 16

AGENT_MAX_ITER_TIMES = 50

PROJECT_ROOT = Path(llm4mdg.__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
CONFIG_LOC = PROJECT_ROOT / 'config.yml'
INTERMEDIATE_DATA_LOC = PROJECT_ROOT / 'intermediates'

DIR_BLACKLIST = [
    '*[Bb]uild*', '*[Dd]atabase*', '*[Ss]tatic*', '*[Tt]est*',
    '*pipeline*', '.circleci', '.git', '.github', '.idea',
    '.ipynb_checkpoints', '.mvn', '.vs', '.vscode', '[Ii]mage',
    '[Ii]mages', '[Tt]emplate*', 'bin', 'node_modules',
    'npm-debug.log', 'obj', "[Pp]ublic", "[Ww]ebroot", "wwwroot",
]

FILE_BLACKLIST = [
    '*.sql', '*lock*', '.*rc', '.*rc.*', '.DS_Store', '.docker*',
    '.editorconfigmvnw*', '.git*', '.prettier*', '.travis.yml',
    'LICENSE', 'gradlew*', 'secrets.dev.yaml', 'tsconfig.*json',
    'values.dev.yaml', ".classpath", "*.html", "*.css", "*.scss",
    "*.js.map", "*.css.map", "*.svg", "*.pem", "*_test.go",
]

VECTORDB_PRIMARY_FIELD = "id"
VECTORDB_TEXT_FIELD = "text"
VECTORDB_VECTOR_FIELD = "vector"

MULTI_THREAD_COUNT = 10
