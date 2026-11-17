# SuperMimic Teleop — entrenamiento en simulación

Añade una entrada humana causal a SuperMimic, inspirada en la interfaz corporal
SMPL de SONIC. El objetivo es entrenar un controlador G1 que siga el cuerpo
humano completo, incluidas las piernas, usando pares SMPL–G1 de BONES-SEED.

**El flujo predeterminado ahora es baseline motion → evaluación → adaptación
humana.** `bash bin/train.sh` entrena únicamente motion. El bootstrap conjunto
anterior sigue disponible con `--stage joint`.

Conservamos `GenCoord`, `actor_obs=rep_state`, historial robot 1 y latent 32.
Todo se ejecuta en MuJoCo; no requiere PICO ni un estimador de estado físico.
El adaptador simulado recibe features humanas ya calibradas. No implementa el
SDK PICO ni un controlador de emergencia para hardware.

La [guía de arquitectura](../GUIDE.md) explica los contratos y fuentes. Para los
comandos actuales, usar este README: las instrucciones anteriores de bootstrap
en la guía preceden a la separación de etapas.

## 1. Instalación y datos

Ejecutar los comandos desde `supermimic-teleop/`. La estructura esperada es:

```text
supermimic-vr/
  supermimic-teleop/
  mjlab_research/
  GR00T-WholeBodyControl/
```

Se necesita GPU NVIDIA compatible con el entorno CUDA/MuJoCo instalado, espacio
para los datos y acceso aceptado a BONES-SEED en Hugging Face. El instalador
prepara Python 3.13 y las dependencias; el preprocesador reutiliza los scripts
del checkout local de GR00T.

En Linux x86_64 se fija PyTorch 2.14.x desde el índice CUDA 12.6 y Warp
1.19.0 desde el wheel oficial `+cu12`. Esto permite usar el driver 570.207
de Razorcrest sin que una sincronización seleccione binarios de CUDA 13.
No hace falta instalar otro CUDA Toolkit del sistema: `uv sync --locked`
instala las dependencias del entorno. PyTorch y Warp deben poder ejecutar
en GPU antes de iniciar el entrenamiento:

```bash
uv run python -c 'import torch, warp as wp; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda")); wp.init(); print(wp.zeros(1, device="cuda:0").numpy())'
```

```bash
bash bin/install.sh
# Si los datos ya están descargados en data/seed, omitir este paso.
bash bin/download_dataset.sh

# Evita necesitar sesión online durante las pruebas.
export WANDB_MODE=offline
export MUJOCO_GL=egl
```

El descargador lee `HF_TOKEN` del entorno, `.env.local` o `.env`, en ese orden.
El entrenamiento admite las mismas fuentes para `WANDB_API_KEY`. La receta
envía los runs a [mu-robotics/supermimic-teleop en Forge](https://forge.coreweave.com/wandb/mu-robotics/supermimic-teleop),
usando `https://api.forge.coreweave.com` como API. En modo online, si faltan
credenciales el entrenamiento se detiene antes de crear el simulador.
El proyecto requiere W&B SDK 0.31.x, con soporte nativo para Forge; `uv sync`
actualiza instalaciones anteriores que pueden fallar durante el login.
Entorno > `.env.local` > `.env` > receta para `WANDB_MODE`, `WANDB_ENTITY` y
`WANDB_BASE_URL`. Las pruebas offline se pueden subir posteriormente con
`WANDB_BASE_URL=https://api.forge.coreweave.com uv run wandb sync --entity mu-robotics --project supermimic-teleop RUTA_DEL_RUN_OFFLINE`.

Para iniciar el baseline con el dataset completo ya descargado:

```bash
# Una sola vez, si todavía no configuraste WANDB_API_KEY para Forge.
uv run wandb login --host https://api.forge.coreweave.com

WANDB_MODE=online bash bin/train.sh --stage baseline \
  --data data/seed-processed-full \
  --output-dir "logs/seed-motion-baseline-$(date +%Y%m%d-%H%M%S)"
```

El lector extrae automáticamente los shards descargados en `.unpacked/` en
el primer uso; no hace falta volver a preprocesarlos. Este comando usa 32
entornos, GPU `cuda:0` y 10 000 iteraciones. Después se evalúa el baseline y
se usa su checkpoint para la adaptación humana (secciones 3–4).

Preparar primero un subconjunto manejable y reproducible:

```bash
bash bin/preprocess_dataset.sh \
  --max-clips 256 --include walk idle jog \
  --output data/seed-sim-256

uv run python -m supermimic.scripts.audit_dataset \
  --data data/seed-sim-256 --plots 12 --strict
```

Se generan `train/`, `validation/`, `manifest.jsonl`, `summary.json` y
`preprocess.json`. La separación es por familia antes de construir ventanas.
El lector remuestrea a 50 Hz, comprueba duración, límites articulares y apoyo FK.
Además, rechaza calibraciones iniciales ambiguas: al menos un punto de pie
inicialmente bajo debe mantenerse a menos de 4 cm de su altura inicial durante
los primeros 120 ms. Es una heurística de apoyo, no una medición de contacto;
puede excluir movimientos válidos. El suelo de las features sigue procediendo
solo del primer frame: la ventana adicional se usa únicamente para aceptar o
rechazar el clip.

La auditoría produce `audit/report.json` y gráficos PNG de altura de pelvis y
yaw humano/G1. Comprueba canales, separación de familias y señala posibles
problemas de heading o desfase temporal. `--strict` termina con error si hay
advertencias; revisar los pares señalados antes de entrenar. La correlación de
altura es orientativa, no determina un desfase fiable en clips estáticos ni
sustituye una inspección de pies/muñecas. No corrige ni desplaza datos en silencio.
Para más gráficos, aumentar `--plots`. Una auditoría sin advertencias no demuestra
alineamiento perfecto.

Para incorporar el nuevo filtro inicial a un dataset anterior, regenerarlo en
otro `--output`. No mezclar recetas. No entrenar mientras exista `.processing`.

### Procesar el corpus completo

`bin/preprocess_dataset.sh` usa por defecto `data/seed` como entrada, los scripts
del checkout hermano `../GR00T-WholeBodyControl` y escribe en `data/seed-processed`.
Para procesar todos los clips que encuentre, desactiva el límite con
`--max-clips 0` y pasa `--include` sin palabras: así se aplica la lista negra
oficial por nombre, pero no se limita a categorías como `walk` o `idle`.

```bash
# Ejecutar desde supermimic-teleop/, después de descargar y extraer los datos.
bash bin/preprocess_dataset.sh \
  --max-clips 0 --include \
  --input data/seed \
  --gr00t ../GR00T-WholeBodyControl \
  --output data/seed-processed-full
```

El proceso busca CSV en `data/seed/g1/csv/` y el `.pkl` SMPL con el mismo
nombre en `data/seed/smpl_filtered/`. Cada pareja se convierte a 50 Hz y pasa
los filtros de nombre, duración, articulaciones, apoyo FK y calibración humana
descritos arriba. Por eso “completo” significa que se recorren todos los
candidatos elegibles; algunos se registran como rechazados o sin pareja, y no
todos terminan en el conjunto aceptado. El comando muestra el resumen al final.
Consulta `summary.json` para los conteos y `manifest.jsonl` para el estado y la
razón de cada candidato que pasó el filtro de nombre. Los nombres excluidos se
cuentan como `name_filter` en el resumen.

El resultado contiene `train/`, `validation/`, `manifest.jsonl`,
`summary.json` y `preprocess.json`. Usa un directorio nuevo para mantener la
salida de un procesamiento anterior. Si el comando se interrumpe, no entrenes:
vuelve a ejecutar el mismo comando para regenerar el corpus completo. El marcador
`.processing` se elimina al terminar correctamente. La receta se guarda en
`preprocess.json`; cambiar opciones requiere otro directorio de salida.

Después, audita el corpus antes de entrenar:

```bash
uv run python -m supermimic.scripts.audit_dataset \
  --data data/seed-processed-full --plots 24 --strict
```

El corpus completo puede ocupar mucho espacio y el entrenamiento carga datos en
memoria del dispositivo. Confirma primero que tienes espacio suficiente y mide
RAM/VRAM antes de lanzar entrenamiento a escala; si no caben, usa un subconjunto
con `--max-clips` o categorías concretas con `--include walk idle jog`.

### Subir los datos procesados a Hugging Face

Para evitar el límite de archivos por directorio del Hub, empaqueta los NPZ ya
procesados en archivos `.tar` de 1 000 clips, agrupados en carpetas de hasta 100 paquetes. **No vuelve a procesar los datos**:
conserva los bytes y las rutas `train/` y `validation/` dentro de cada paquete.
La salida ocupa aproximadamente otros 25 GB para este corpus; usa una carpeta
nueva fuera del dataset de entrada.

```bash
python bin/package_dataset.py \
  --input data/seed-processed-full \
  --output data/seed-processed-full-hub

hf upload NONHUMAN-RESEARCH/seed-processed-full \
  ./data/seed-processed-full-hub . --repo-type dataset \
  --exclude ".unpacked/**" --exclude ".cache/**"
```

Se suben `shards/*.tar`, un índice `shards.json` con conteos y SHA-256, los tres
archivos de metadatos y un README con instrucciones de descarga. Si la subida
se interrumpe, repite el comando `hf upload`. Los archivos sueltos de un intento
anterior pueden seguir en el remoto; la descarga de abajo selecciona únicamente
los paquetes y metadatos nuevos.

El preprocesamiento ahora empaqueta automáticamente al terminar, en `<output>-hub`.
Para empaquetar datos existentes sin ejecutar ninguna conversión:

```bash
bin/preprocess_dataset.sh --package-only --output data/seed-processed-full \
  --hub-output data/seed-processed-full-hub-nuevo
```

`--clips-per-shard` controla el número de clips por paquete. El destino debe ser
nuevo para evitar mezclar versiones. Entrenamiento y auditoría aceptan la raíz
del paquete mediante `--data`; evaluación acepta esa raíz (elige validation)
o su ruta `validation/`. En el primer uso se comprueban SHA-256 y conteos y se
extraen ambos splits en `.unpacked/`; los siguientes usos reutilizan esa copia.
Se necesita espacio adicional para los NPZ extraídos. Por ejemplo:

```bash
bin/train.sh --data data/seed-processed-full-hub
uv run python -m supermimic.scripts.evaluate_human \
  --checkpoint logs/my-run --data data/seed-processed-full-hub/validation
```

Para usarlo en otra máquina, descarga y extrae los paquetes (no requiere
reprocesamiento). Ejecuta desde `supermimic-teleop/`:

```bash
hf download NONHUMAN-RESEARCH/seed-processed-full --repo-type dataset \
  --include 'shards/*' --include 'shards.json' --include 'manifest.jsonl' \
  --include 'preprocess.json' --include 'summary.json' \
  --local-dir data/seed-processed-full-download

mkdir -p data/seed-processed-full
for shard in data/seed-processed-full-download/shards/*/*/*.tar; do
  tar -xf "$shard" -C data/seed-processed-full
done
cp data/seed-processed-full-download/{manifest.jsonl,preprocess.json,summary.json} \
  data/seed-processed-full/
```

Tras extraer, los comandos de entrenamiento y auditoría usan
`--data data/seed-processed-full` como con el dataset original. Las rutas de
origen guardadas en el manifest y la receta describen la máquina que procesó
los datos y se mantienen como procedencia.

## 2. Verificar el recorrido con una prueba pequeña

Estos comandos prueban preparación, baseline, carga, adaptación y exportación.
Los directorios de salida deben ser nuevos; se rechaza sobrescribirlos.

```bash
bash bin/preprocess_dataset.sh --output data/seed-sim-smoke

bash bin/train.sh --stage baseline --smoke \
  --data data/seed-sim-smoke --output-dir logs/smoke-motion

bash bin/train.sh --stage adaptation --distillation decay --smoke \
  --checkpoint logs/smoke-motion \
  --data data/seed-sim-smoke --output-dir logs/smoke-human

uv run python -m supermimic.scripts.evaluate_human \
  --checkpoint logs/smoke-human --data data/seed-sim-smoke/validation \
  --steps 10 --wandb-mode disabled --output reports/smoke-human.json
```

`--smoke` usa 4 entornos, redes 64→32 y 8 iteraciones. Sus pesos **no son un
teacher útil ni un controlador entrenado**. No cargar esos pesos en la receta
completa: tamaños de red, representación e historial deben coincidir.
`--steps 10` recorre todos los clips, pero limita cada ensayo y lo marca como
`step_limit` si no terminó antes; no equivale a completar el clip.

## 3. Entrenar y evaluar el baseline motion

```bash
# Inspeccionar la configuración efectiva sin ejecutar.
bash bin/train.sh --stage baseline --data data/seed-sim-256 --print-config

# Arquitectura completa, 32 entornos y 10 000 iteraciones por defecto.
bash bin/train.sh --stage baseline \
  --data data/seed-sim-256 --output-dir logs/sim-motion-baseline

# Sin --steps: cada ensayo continúa hasta final de clip o fallo.
uv run python -m supermimic.scripts.evaluate_human \
  --checkpoint logs/sim-motion-baseline --branch motion \
  --data data/seed-sim-256/validation --num-envs 4 \
  --output reports/baseline-motion.json
```

El evaluador selecciona explícitamente **cada clip** desde su primer frame.
`--num-envs 4` ejecuta cuatro ensayos por clip con la randomización del entorno.
Registra por ensayo duración, reward, error corporal, pelvis, muñecas y velocidad
horizontal de pies bajos como proxy de deslizamiento (no fuerza de contacto).
Clasifica `completed`, `failure`, `time_limit`, `step_limit` y `sensor_timeout`.
Un fallo físico en el último frame sigue contando como fallo. Los promedios de
error incluyen solo el tiempo observado, por lo que siempre deben leerse junto
con tasa de finalización y duración.

Revisar tasa de clips completados, fallos y distribución por familia antes de
usar el baseline como teacher. El número de iteraciones no garantiza calidad;
si motion falla sistemáticamente, mejorar primero ese baseline. El programa no
certifica automáticamente que un checkpoint sea bueno por haber sido evaluado.

## 4. Adaptar el encoder humano y comparar distilación

Cada experimento empieza desde **el mismo baseline**. Se congelan encoder
motion/velocity, decoder y normalizadores originales; se entrena el encoder y
normalizador humano. El WM sigue ajustándose con transiciones del alumno y queda
congelado respecto a parámetros durante BPTT, conservando gradientes a acciones.
La carga es un warm start de pesos, no una reanudación exacta de replay/optimizadores.

```bash
# Tres experimentos independientes y comparables.
for mode in off fixed decay; do
  bash bin/train.sh --stage adaptation --distillation "$mode" \
    --checkpoint logs/sim-motion-baseline \
    --data data/seed-sim-256 --output-dir "logs/sim-human-$mode"
done

for mode in off fixed decay; do
  uv run python -m supermimic.scripts.evaluate_human \
    --checkpoint "logs/sim-human-$mode" --branch human_body \
    --data data/seed-sim-256/validation --num-envs 4 \
    --output "reports/human-$mode-clean.json"
done

uv run python -m supermimic.scripts.compare_evaluations \
  reports/baseline-motion.json \
  reports/human-off-clean.json reports/human-fixed-clean.json \
  reports/human-decay-clean.json
```

| Modo | Pérdidas auxiliares |
|---|---|
| `off` | Solo objetivo físico; pesos latent/acción = 0 |
| `fixed` | MSE latent × 0,1 + MSE acción × 1,0 durante toda la adaptación |
| `decay` | Mismos pesos iniciales, multiplicados por `max(0, 1 - iteración / 5000)` |

El teacher motion ve hasta 900 ms de futuro; el humano solo ve pasado/presente.
Por eso la distilación es un auxiliar a contrastar con `off`, no un target causal
perfecto. Comparar especialmente arranques, frenadas y cambios de apoyo.

En la receta de 10 000 iteraciones, `decay` llega a cero en la iteración 5000.
Para otra duración, especificar también el punto de fin:

```bash
bash bin/train.sh --stage adaptation --distillation decay \
  --checkpoint logs/sim-motion-baseline --data data/seed-sim-256 \
  --output-dir logs/sim-human-decay-long \
  --set runner.max_iterations=20000 --set runner.bc_decay_iterations=10000
```

El factor es un tensor actualizado in-place para ser compatible con CUDA graphs.
W&B registra `Policy/bc_scale`. En `joint`, la pérdida de acciones también puede
actualizar el decoder; en `adaptation` este permanece congelado.

Para comprobar también la rama original después de adaptar:

```bash
uv run python -m supermimic.scripts.evaluate_human \
  --checkpoint logs/sim-human-decay --branch motion \
  --data data/seed-sim-256/validation --num-envs 4 \
  --output reports/adapted-motion.json

uv run python -m supermimic.scripts.compare_evaluations \
  reports/baseline-motion.json reports/adapted-motion.json
```

## 5. Probar teleoperación simulada: retardo, pérdidas y caducidad

La evaluación humana usa `HumanStream`: recibe paquetes calibrados con timestamp
de captura, descarta paquetes duplicados/atrasados, retiene la estimación más
reciente y construye historia a 50 Hz. No interpola con muestras futuras.
Actualiza edad incluso sin recibir paquetes; tras superar `--max-age-ms`, termina
el ensayo como `sensor_timeout`, sin seguir aplicando un comando antiguo.
Al reiniciar/recentrar se debe llamar a `reset()`. La pose inicial calibrada está
disponible antes de iniciar cada ensayo; las perturbaciones se aplican después.

```bash
# Mismo escenario degradado para las tres políticas.
for mode in off fixed decay; do
  uv run python -m supermimic.scripts.evaluate_human \
    --checkpoint "logs/sim-human-$mode" \
    --data data/seed-sim-256/validation --num-envs 4 \
    --delay-ms 40 --jitter-ms 20 --position-noise 0.005 \
    --dropout 0.02 --max-dropout-frames 4 --max-age-ms 200 \
    --output "reports/human-$mode-delayed.json"
done

uv run python -m supermimic.scripts.compare_evaluations \
  reports/human-off-delayed.json reports/human-fixed-delayed.json \
  reports/human-decay-delayed.json

# Interrumpir paquetes desde t=0,5 s durante 0,5 s.
uv run python -m supermimic.scripts.evaluate_human \
  --checkpoint logs/sim-human-decay \
  --data data/seed-sim-256/validation --num-envs 4 \
  --outage-at 0.5 --outage-seconds 0.5 --max-age-ms 200 \
  --output reports/human-decay-outage.json
```

`jitter` añade un retardo uniforme entre 0 y el valor indicado. El dropout global
retiene la última estimación y propaga flags/edad. La simulación trabaja a pasos
de 20 ms; no pretende medir latencia del SDK, red o hardware. Los cuatro ensayos
de un clip comparten la misma señal humana perturbada. Un fallo físico puede
ocurrir antes del timeout; ambos motivos se registran por separado.

`compare_evaluations` exige los mismos clips, ensayos, seed, límite de pasos y
parámetros de sensor: comparar políticas dentro de cada escenario. Para comparar
limpio frente a degradado, consultar sus informes por separado. Reservar sesiones
PICO y validar calibración real queda para una fase posterior.

## Configuración, artefactos y escala

La configuración base está en [configs/train.yaml](configs/train.yaml).
`--stage` aplica el preset sobre el YAML; `--smoke` reduce tamaños; los `--set`
se aplican al final. `--data` recibe la raíz procesada, mientras `evaluate_human
--data` recibe específicamente `validation/`. Todas las rutas son relativas a
este proyecto; `bin/train.sh` cambia automáticamente a su directorio.

Cada salida contiene `runner_config.json`, `actor.pt`, `world_model.pt`,
`critic.pt` y, en la etapa humana, `policy_human_body.onnx`. El ONNX acepta
`goal_human [B,1360]` y `history_robot [B,1,68]`, y devuelve acciones `[B,29]`.
No contiene world model ni clips. La escala/offset/PD sigue siendo responsabilidad
del entorno de ejecución; esta tarea mantiene el despliegue en simulación.

Para ajustar recursos, usar por ejemplo `--set num_envs=16
--set runner.policy.batch_size=32 --set runner.wm.batch_size=32`.
`runtime.compile` y `runtime.cuda_graphs` permanecen desactivados por defecto;
se pueden activar con `--set` tras verificar una ejecución pequeña. Cambiar las
capas del actor/WM o el historial exige un baseline compatible.

El loader actual carga las referencias y tablas humanas en memoria del
dispositivo. Este cambio no implementa shards. La selección walk/idle/jog
inicial no acredita generalización a todo el cuerpo.

Para reproducir el bootstrap conjunto experimental anterior:

```bash
bash bin/train.sh --stage joint --distillation fixed \
  --data data/seed-sim-256 --output-dir logs/sim-joint
```

Comprobaciones de código:

```bash
uv run pytest -q
uv run ruff check src tests
```

Verificación de esta implementación: 232 pruebas pasan; Ruff, sintaxis shell y
consistencia del lockfile pasan. Se ejecutaron preprocesado de 16 clips,
baseline/adaptación GPU de integración, evaluación de los 5 clips reservados y
un corte de señal que produjo `sensor_timeout` en los 5. Los tensores originales
del actor quedaron idénticos después de adaptar. Esto verifica el recorrido;
los checkpoints de prueba siguen fallando físicamente y no son un baseline útil.

## Referencias y licencia

[SuperMimic](https://supermimic.github.io) ·
[Paper](https://arxiv.org/abs/2610.28480) ·
[SONIC local](../GR00T-WholeBodyControl/README.md).

Basado en SuperMimic y mjlab, con código inspirado en ControlVAE,
robotic_world_model y DreamerV3. Copyright 2026 The SuperMimic Authors.
Licencia [Apache-2.0](LICENSE).
