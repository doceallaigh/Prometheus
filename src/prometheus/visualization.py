from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from prometheus.config import PrometheusConfig
from prometheus.model import StaticRouter, resolve_modular_layout, resolve_modular_stage_specs


def build_structure_payload(config: PrometheusConfig) -> dict[str, Any]:
    """Build a serializable structural view of a model configuration.

    Args:
        config: Experiment configuration containing the model layout.

    Returns:
        dict[str, Any]: Nodes, edges, and metadata for visualization.
    """

    model = config.model
    payload: dict[str, Any] = {
        "run_name": config.experiment.run_name,
        "architecture": model.architecture,
        "embedding_dim": model.embedding_dim,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "mlp_ratio": model.mlp_ratio,
        "sequence_length": config.data.sequence_length,
        "nodes": [],
        "edges": [],
        "routing_matrices": [],
        "stage_profiles": [],
        "model_config": asdict(model),
    }
    payload["nodes"].append(
        {
            "id": "input",
            "label": "Input",
            "kind": "io",
            "stage_index": -1,
            "group_index": -1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "details": f"sequence_length={config.data.sequence_length}",
        }
    )

    unit_label = "Modular group"
    if model.architecture == "dense":
      _add_dense_structure(payload, model.embedding_dim, model.num_layers)
    elif model.architecture in {"modular", "cortical_columns"}:
      layout = resolve_modular_layout(model)
      unit_label = layout.unit_label
      _add_modular_structure(payload, resolve_modular_stage_specs(model), layout.routing_topology, layout.routing_top_k)
    else:
        raise ValueError(f"Unsupported architecture: {model.architecture}")
    payload["unit_label"] = unit_label

    payload["nodes"].append(
        {
            "id": "output",
            "label": "Output",
            "kind": "io",
            "stage_index": len(payload.get("stages", [])),
            "group_index": -1,
            "x": float(len(payload.get("stages", [])) + 1),
            "y": 0.0,
            "z": 0.0,
            "details": f"vocab_size={model.vocab_size}",
        }
    )
    _connect_last_stage_to_output(payload)
    return payload


def render_structure_html(config: PrometheusConfig) -> str:
    """Render an interactive HTML page for the configured model structure."""

    payload = build_structure_payload(config)
    payload_json = json.dumps(payload)
    preview_layout = resolve_modular_layout(config.model) if config.model.architecture in {"modular", "cortical_columns"} else None
    group_label = "Column counts" if config.model.architecture == "cortical_columns" else "Stage groups"
    depth_label = "Column depths" if config.model.architecture == "cortical_columns" else "Stage depths"
    fixed_label = "Fixed column size" if config.model.architecture == "cortical_columns" else "Fixed group size"
    preview_groups = [] if preview_layout is None else preview_layout.group_schedule
    preview_depths = [] if preview_layout is None else preview_layout.depth_schedule
    preview_fixed_size = "" if preview_layout is None or preview_layout.fixed_group_size is None else preview_layout.fixed_group_size
    preview_routing_topology = config.model.routing_topology if preview_layout is None else preview_layout.routing_topology
    preview_routing_top_k = "" if preview_layout is None or preview_layout.routing_top_k is None else preview_layout.routing_top_k
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Prometheus Model Structure</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1220;
      --panel: #111a2b;
      --text: #e7edf7;
      --muted: #9fb0c8;
      --dense: #f6bd60;
      --module: #84dcc6;
      --route: #f28482;
      --io: #cdb4db;
      --accent: #90be6d;
    }}
    body {{
      margin: 0;
      font-family: Segoe UI, Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at top, #162238, var(--bg) 60%);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      min-height: 100vh;
    }}
    .panel {{
      padding: 24px;
      background: linear-gradient(180deg, rgba(17, 26, 43, 0.98), rgba(11, 18, 32, 0.96));
      border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    h1 {{
      font-size: 1.4rem;
      margin: 0 0 8px;
    }}
    h2 {{
      font-size: 1rem;
      margin: 24px 0 8px;
      color: var(--dense);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .pill {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.07);
      font-size: 0.85rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 8px 0;
    }}
    .control-grid {{
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }}
    .control-grid.two-col {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .control-field {{
      display: grid;
      gap: 6px;
    }}
    .control-field label,
    .toggle {{
      color: var(--text);
      font-size: 0.85rem;
    }}
    .control-field input,
    .control-field select {{
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .action-button {{
      border: 0;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .action-button.primary {{
      background: rgba(144, 190, 109, 0.2);
      box-shadow: inset 0 0 0 1px rgba(144, 190, 109, 0.35);
      color: #f4ffe7;
    }}
    .helper-text {{
      margin: 0;
      font-size: 0.8rem;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(255, 255, 255, 0.04);
      padding: 12px;
      border-radius: 12px;
      color: var(--text);
      font-size: 0.8rem;
      overflow: auto;
    }}
    #plot {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    .view-switcher {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 20px 20px 0;
    }}
    .view-button {{
      border: 0;
      background: rgba(255, 255, 255, 0.07);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 0.9rem;
    }}
    .view-button.active {{
      background: rgba(144, 190, 109, 0.2);
      box-shadow: inset 0 0 0 1px rgba(144, 190, 109, 0.35);
      color: #f4ffe7;
    }}
    .plot-surface {{
      min-height: 0;
      height: calc(100vh - 82px);
      display: none;
    }}
    .plot-surface.active {{
      display: block;
    }}
    @media (max-width: 960px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .panel {{
        border-right: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }}
      #plot {{
        min-height: 70vh;
      }}
      .plot-surface {{
        height: 70vh;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"panel\">
      <h1>{config.experiment.run_name}</h1>
      <p>Interactive structural view generated directly from the config and the router mask implementation. Use this to verify stage grouping and branching paths.</p>
      <div class=\"pill-row\">
        <span class=\"pill\">architecture: {config.model.architecture}</span>
        <span class=\"pill\">embedding_dim: {config.model.embedding_dim}</span>
        <span class=\"pill\">num_heads: {config.model.num_heads}</span>
        <span class=\"pill\">seq_len: {config.data.sequence_length}</span>
      </div>
      <h2>Legend</h2>
      <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--io);\"></span><span>Input / output</span></div>
      <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--dense);\"></span><span>Dense block</span></div>
      <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--module);\"></span><span>{payload['unit_label']}</span></div>
      <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--route);\"></span><span>Allowed routing branch</span></div>
      <h2>Notes</h2>
      <ul>
        <li>X axis shows progression through stages or blocks.</li>
        <li>Y axis separates groups within a stage.</li>
        <li>Z axis is used to lift routing edges away from local block edges for easier inspection.</li>
        <li>The adjacency view is the most exact way to verify allowed branching paths.</li>
        <li>The stage profile view summarizes branching pressure without adding more geometry to the 3D scene.</li>
      </ul>
      <h2>Preview Controls</h2>
      <div class="control-grid two-col">
        <div class="control-field">
          <label for="stage-groups-input">{group_label}</label>
          <input id="stage-groups-input" type="text" value="{','.join(str(value) for value in preview_groups)}" placeholder="4,2,1" />
        </div>
        <div class="control-field">
          <label for="stage-depths-input">{depth_label}</label>
          <input id="stage-depths-input" type="text" value="{','.join(str(value) for value in preview_depths)}" placeholder="1,1,1" />
        </div>
        <div class="control-field">
          <label for="fixed-group-size-input">{fixed_label}</label>
          <input id="fixed-group-size-input" type="number" min="1" step="1" value="{preview_fixed_size}" placeholder="optional" />
        </div>
        <div class="control-field">
          <label for="routing-topology-select">Routing topology</label>
          <select id="routing-topology-select">
            <option value="dense"{' selected' if preview_routing_topology == 'dense' else ''}>dense</option>
            <option value="local"{' selected' if preview_routing_topology == 'local' else ''}>local</option>
            <option value="small_world"{' selected' if preview_routing_topology == 'small_world' else ''}>small_world</option>
            <option value="cluster_graph"{' selected' if preview_routing_topology == 'cluster_graph' else ''}>cluster_graph</option>
          </select>
        </div>
        <div class="control-field">
          <label for="routing-top-k-input">Routing top-k cap</label>
          <input id="routing-top-k-input" type="number" min="1" step="1" value="{preview_routing_top_k}" placeholder="optional" />
        </div>
      </div>
      <div class="button-row">
        <button id="apply-preview" class="action-button primary" type="button">Apply Preview</button>
        <button id="reset-preview" class="action-button" type="button">Reset Preview</button>
      </div>
      <p class="helper-text">These controls rebuild the visualization in the browser. They do not change training configs on disk.</p>
      <p class="helper-text">The top-k value caps how many inbound routes can be kept per group during routing, but it does not identify a single fixed subset of edges without learned routing scores.</p>
      <h2>View Controls</h2>
      <div class="control-grid">
        <div class="control-field">
          <label for="stage-focus-select">Focus stage</label>
          <select id="stage-focus-select"></select>
        </div>
        <label class="toggle"><input id="show-flow-toggle" type="checkbox" checked /> Show cross-stage flow edges</label>
        <label class="toggle"><input id="show-self-routes-toggle" type="checkbox" checked /> Show self-routing edges</label>
      </div>
      <h2>Model Config</h2>
      <pre id="model-config">{json.dumps(asdict(config.model), indent=2)}</pre>
    </aside>
    <main id=\"plot\">
      <div class=\"view-switcher\">
        <button class=\"view-button active\" data-target=\"structure-plot\">3D Structure</button>
        <button class=\"view-button\" data-target=\"routing-plot\">Routing Matrix</button>
        <button class=\"view-button\" data-target=\"profile-plot\">Stage Profile</button>
      </div>
      <div id=\"structure-plot\" class=\"plot-surface active\"></div>
      <div id=\"routing-plot\" class=\"plot-surface\"></div>
      <div id=\"profile-plot\" class=\"plot-surface\"></div>
    </main>
  </div>
  <script>
    const originalPayload = {payload_json};
    let payload = cloneValue(originalPayload);
    const kindColors = {{ io: '#cdb4db', dense: '#f6bd60', module: '#84dcc6' }};

    function cloneValue(value) {{
      return JSON.parse(JSON.stringify(value));
    }}

    function nodeMapFor(activePayload) {{
      return new Map(activePayload.nodes.map((node) => [node.id, node]));
    }}

    function parseIntegerList(text, fallback) {{
      const raw = text.trim();
      if (!raw) {{
        return fallback;
      }}
      const values = raw.split(',').map((part) => Number.parseInt(part.trim(), 10));
      if (values.some((value) => !Number.isInteger(value) || value <= 0)) {{
        throw new Error('Expected a comma-separated list of positive integers.');
      }}
      return values;
    }}

    function buildRoutingMask(numGroups, topology) {{
      const mask = Array.from({{ length: numGroups }}, () => Array.from({{ length: numGroups }}, () => 0));
      const clusterSize = Math.max(2, Math.floor(numGroups ** 0.5));
      for (let destination = 0; destination < numGroups; destination += 1) {{
        for (let source = 0; source < numGroups; source += 1) {{
          let allowed;
          if (topology === 'dense') {{
            allowed = true;
          }} else if (topology === 'local') {{
            allowed = source === destination || Math.abs(source - destination) === 1;
          }} else if (topology === 'small_world') {{
            const stride = Math.max(1, Math.floor(numGroups / 2));
            allowed = source === destination || Math.abs(source - destination) === 1 || source === (destination + stride) % numGroups;
          }} else if (topology === 'cluster_graph') {{
            const destinationCluster = Math.floor(destination / clusterSize);
            const sourceCluster = Math.floor(source / clusterSize);
            allowed = sourceCluster === destinationCluster;
            if (!allowed) {{
              const clusterCount = Math.max(1, Math.ceil(numGroups / clusterSize));
              allowed = source === ((destinationCluster + 1) % clusterCount) * clusterSize;
            }}
          }} else {{
            throw new Error(`Unsupported routing topology: ${{topology}}`);
          }}
          if (allowed) {{
            mask[destination][source] = 1;
          }}
        }}
      }}
      return mask;
    }}

    function appendEdge(activePayload, nodesById, sourceId, targetId, kind, options = {{}}) {{
      const source = nodesById.get(sourceId);
      const target = nodesById.get(targetId);
      activePayload.edges.push({{
        source: sourceId,
        target: targetId,
        kind,
        details: options.details || '',
        stage_index: options.stageIndex ?? null,
        x0: source.x,
        y0: source.y,
        z0: source.z + (options.zOffset || 0),
        x1: target.x,
        y1: target.y,
        z1: target.z + (options.zOffset || 0),
      }});
    }}

    function resolveStageSpecs(modelConfig) {{
      const usesColumns = modelConfig.architecture === 'cortical_columns';
      const groupSchedule = usesColumns
        ? ((modelConfig.column_counts && modelConfig.column_counts.length) ? modelConfig.column_counts : (() => {{ throw new Error('cortical_columns requires column_counts.'); }})())
        : ((modelConfig.stage_groups && modelConfig.stage_groups.length) ? modelConfig.stage_groups : [2, 1]);
      const depthSchedule = usesColumns
        ? ((modelConfig.column_depths && modelConfig.column_depths.length) ? modelConfig.column_depths : Array.from({{ length: groupSchedule.length }}, () => 1))
        : ((modelConfig.stage_depths && modelConfig.stage_depths.length) ? modelConfig.stage_depths : Array.from({{ length: groupSchedule.length }}, () => 1));
      if (groupSchedule.length !== depthSchedule.length) {{
        throw new Error('stage_groups and stage_depths must have the same length.');
      }}
      const fixedGroupSize = usesColumns ? modelConfig.fixed_column_size : modelConfig.fixed_group_size;
      if (fixedGroupSize != null) {{
        if (!Number.isInteger(fixedGroupSize) || fixedGroupSize <= 0) {{
          throw new Error('fixed_group_size must be a positive integer when provided.');
        }}
      }}
      return groupSchedule.map((groupCount, index) => {{
        if (!Number.isInteger(groupCount) || groupCount <= 0) {{
          throw new Error('stage_groups must contain positive integers.');
        }}
        const depth = depthSchedule[index];
        const groupDim = fixedGroupSize == null ? (() => {{
          if (modelConfig.embedding_dim % groupCount !== 0) {{
            throw new Error('embedding_dim must be divisible by each stage group count.');
          }}
          return Math.floor(modelConfig.embedding_dim / groupCount);
        }})() : fixedGroupSize;
        return {{
          groupCount,
          depth,
          groupDim,
          stageDim: groupCount * groupDim,
        }};
      }});
    }}

    function buildPayloadFromModelConfig(modelConfig) {{
      const activePayload = {{
        run_name: originalPayload.run_name,
        architecture: modelConfig.architecture,
        embedding_dim: modelConfig.embedding_dim,
        num_heads: modelConfig.num_heads,
        num_layers: modelConfig.num_layers,
        mlp_ratio: modelConfig.mlp_ratio,
        sequence_length: originalPayload.sequence_length,
        nodes: [],
        edges: [],
        routing_matrices: [],
        stage_profiles: [],
        model_config: cloneValue(modelConfig),
        stages: [],
      }};
      activePayload.nodes.push({{
        id: 'input',
        label: 'Input',
        kind: 'io',
        stage_index: -1,
        group_index: -1,
        x: 0,
        y: 0,
        z: 0,
        details: `sequence_length=${{originalPayload.sequence_length}}`,
      }});

      if (modelConfig.architecture === 'dense') {{
        let previousId = 'input';
        for (let index = 0; index < modelConfig.num_layers; index += 1) {{
          const nodeId = `dense-${{index}}`;
          activePayload.stages.push({{ label: `Dense block ${{index + 1}}` }});
          activePayload.stage_profiles.push({{
            label: `Dense ${{index + 1}}`,
            group_count: 1,
            group_dim: modelConfig.embedding_dim,
            depth: 1,
            allowed_routes: 1,
            max_inbound_routes: 1,
            effective_inbound_cap: 1,
          }});
          activePayload.nodes.push({{
            id: nodeId,
            label: `Dense ${{index + 1}}`,
            kind: 'dense',
            stage_index: index,
            group_index: 0,
            x: index + 1,
            y: 0,
            z: 0,
            details: `embedding_dim=${{modelConfig.embedding_dim}}`,
          }});
          const nodesById = nodeMapFor(activePayload);
          appendEdge(activePayload, nodesById, previousId, nodeId, 'flow', {{ stageIndex: index }});
          previousId = nodeId;
        }}
      }} else if (modelConfig.architecture === 'modular' || modelConfig.architecture === 'cortical_columns') {{
        const stageSpecs = resolveStageSpecs(modelConfig);
        const routingTopology = modelConfig.architecture === 'cortical_columns'
          ? (modelConfig.column_routing_topology || modelConfig.routing_topology)
          : modelConfig.routing_topology;
        const routingTopK = modelConfig.architecture === 'cortical_columns'
          ? (modelConfig.column_routing_top_k == null ? modelConfig.routing_top_k : modelConfig.column_routing_top_k)
          : modelConfig.routing_top_k;
        let previousStageNodes = ['input'];
        for (let stageIndex = 0; stageIndex < stageSpecs.length; stageIndex += 1) {{
          const stageSpec = stageSpecs[stageIndex];
          const groupCount = stageSpec.groupCount;
          const depth = stageSpec.depth;
          const groupDim = stageSpec.groupDim;
          activePayload.stages.push({{ label: `Stage ${{stageIndex + 1}}`, group_count: groupCount, depth }});
          const routingMask = buildRoutingMask(groupCount, routingTopology);
          const inboundCounts = routingMask.map((row) => row.reduce((total, value) => total + value, 0));
          const allowedRoutes = inboundCounts.reduce((total, value) => total + value, 0);
          const maxInboundRoutes = inboundCounts.length ? Math.max(...inboundCounts) : 0;
          activePayload.stage_profiles.push({{
            label: `Stage ${{stageIndex + 1}}`,
            group_count: groupCount,
            group_dim: groupDim,
            stage_dim: stageSpec.stageDim,
            depth,
            allowed_routes: allowedRoutes,
            max_inbound_routes: maxInboundRoutes,
            effective_inbound_cap: routingTopK == null ? maxInboundRoutes : Math.min(maxInboundRoutes, routingTopK),
          }});

          const currentStageNodes = [];
          const yCenter = (groupCount - 1) / 2;
          for (let groupIndex = 0; groupIndex < groupCount; groupIndex += 1) {{
            const nodeId = `stage-${{stageIndex}}-group-${{groupIndex}}`;
            activePayload.nodes.push({{
              id: nodeId,
              label: `S${{stageIndex + 1}}G${{groupIndex + 1}}`,
              kind: 'module',
              stage_index: stageIndex,
              group_index: groupIndex,
              x: stageIndex + 1,
              y: groupIndex - yCenter,
              z: 0,
              details: `group_dim=${{groupDim}}, depth=${{depth}}`,
            }});
            currentStageNodes.push(nodeId);
          }}

          let nodesById = nodeMapFor(activePayload);
          for (const nodeId of currentStageNodes) {{
            for (const previousId of previousStageNodes) {{
              appendEdge(activePayload, nodesById, previousId, nodeId, 'flow', {{ stageIndex }});
            }}
          }}
          activePayload.routing_matrices.push({{
            label: `Stage ${{stageIndex + 1}}`,
            labels: Array.from({{ length: groupCount }}, (_, groupIndex) => `G${{groupIndex + 1}}`),
            values: routingMask,
          }});
          nodesById = nodeMapFor(activePayload);
          for (let destination = 0; destination < groupCount; destination += 1) {{
            for (let source = 0; source < groupCount; source += 1) {{
              if (!routingMask[destination][source]) {{
                continue;
              }}
              appendEdge(activePayload, nodesById, `stage-${{stageIndex}}-group-${{source}}`, `stage-${{stageIndex}}-group-${{destination}}`, 'route', {{
                details: `topology=${{routingTopology}}, top_k=${{routingTopK ?? 'none'}}`,
                zOffset: 0.9,
                stageIndex,
              }});
            }}
          }}
          previousStageNodes = currentStageNodes;
        }}
      }} else {{
        throw new Error(`Unsupported architecture: ${{modelConfig.architecture}}`);
      }}

      activePayload.nodes.push({{
        id: 'output',
        label: 'Output',
        kind: 'io',
        stage_index: activePayload.stages.length,
        group_index: -1,
        x: activePayload.stages.length + 1,
        y: 0,
        z: 0,
        details: `vocab_size=${{modelConfig.vocab_size}}`,
      }});
      const nodesById = nodeMapFor(activePayload);
      if (activePayload.architecture === 'dense') {{
        if (modelConfig.num_layers === 0) {{
          appendEdge(activePayload, nodesById, 'input', 'output', 'flow', {{ stageIndex: 0 }});
        }} else {{
          appendEdge(activePayload, nodesById, `dense-${{modelConfig.num_layers - 1}}`, 'output', 'flow', {{ stageIndex: modelConfig.num_layers }});
        }}
      }} else {{
        const stageSpecs = resolveStageSpecs(modelConfig);
        const finalStageIndex = stageSpecs.length - 1;
        const finalGroupCount = stageSpecs[finalStageIndex].groupCount;
        for (let groupIndex = 0; groupIndex < finalGroupCount; groupIndex += 1) {{
          appendEdge(activePayload, nodesById, `stage-${{finalStageIndex}}-group-${{groupIndex}}`, 'output', 'flow', {{ stageIndex: finalStageIndex + 1 }});
        }}
      }}
      return activePayload;
    }}

    function focusedNodes(activePayload, focusStage) {{
      if (focusStage === 'all') {{
        return activePayload.nodes;
      }}
      const numericStage = Number.parseInt(focusStage, 10);
      return activePayload.nodes.filter((node) => node.kind === 'io' || node.stage_index === numericStage);
    }}

    function focusedEdges(activePayload, focusStage, showFlowEdges, showSelfRoutes) {{
      const numericStage = focusStage === 'all' ? null : Number.parseInt(focusStage, 10);
      return activePayload.edges.filter((edge) => {{
        if (edge.kind === 'flow' && !showFlowEdges) {{
          return false;
        }}
        if (edge.kind === 'route' && !showSelfRoutes && edge.source === edge.target) {{
          return false;
        }}
        if (numericStage == null) {{
          return true;
        }}
        const sourceNode = activePayload.nodes.find((node) => node.id === edge.source);
        const targetNode = activePayload.nodes.find((node) => node.id === edge.target);
        return (sourceNode && (sourceNode.kind === 'io' || sourceNode.stage_index === numericStage)) || (targetNode && (targetNode.kind === 'io' || targetNode.stage_index === numericStage));
      }});
    }}

    function edgeTrace(activePayload, edgeKind, color, width, focusStage, showFlowEdges, showSelfRoutes) {{
      const xs = [];
      const ys = [];
      const zs = [];
      for (const edge of focusedEdges(activePayload, focusStage, showFlowEdges, showSelfRoutes).filter((item) => item.kind === edgeKind)) {{
        xs.push(edge.x0, edge.x1, null);
        ys.push(edge.y0, edge.y1, null);
        zs.push(edge.z0, edge.z1, null);
      }}
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: xs,
        y: ys,
        z: zs,
        hoverinfo: 'none',
        line: {{ color, width }},
        name: edgeKind,
      }};
    }}

    function nodeTrace(activePayload, kind, focusStage) {{
      const nodes = focusedNodes(activePayload, focusStage).filter((node) => node.kind === kind);
      return {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: nodes.map((node) => node.x),
        y: nodes.map((node) => node.y),
        z: nodes.map((node) => node.z),
        text: nodes.map((node) => node.label),
        textposition: 'top center',
        hovertemplate: nodes.map((node) => `<b>${{node.label}}</b><br>${{node.details}}<extra></extra>`),
        marker: {{
          size: kind === 'io' ? 8 : 6,
          color: kindColors[kind],
          opacity: 0.95,
        }},
        name: kind,
      }};
    }}

    const structureLayout = {{
      paper_bgcolor: '#0b1220',
      plot_bgcolor: '#0b1220',
      margin: {{ l: 0, r: 0, t: 0, b: 0 }},
      showlegend: true,
      legend: {{ font: {{ color: '#e7edf7' }} }},
      scene: {{
        bgcolor: '#0b1220',
        xaxis: {{ title: 'stage / block', color: '#9fb0c8', gridcolor: 'rgba(255,255,255,0.08)' }},
        yaxis: {{ title: 'group index', color: '#9fb0c8', gridcolor: 'rgba(255,255,255,0.08)' }},
        zaxis: {{ title: 'routing depth', color: '#9fb0c8', gridcolor: 'rgba(255,255,255,0.08)' }},
        camera: {{ eye: {{ x: 1.9, y: 1.7, z: 1.2 }} }},
      }},
    }};

    function structureTraces(activePayload) {{
      const focusStage = document.getElementById('stage-focus-select').value || 'all';
      const showFlowEdges = document.getElementById('show-flow-toggle').checked;
      const showSelfRoutes = document.getElementById('show-self-routes-toggle').checked;
      return [
        edgeTrace(activePayload, 'flow', 'rgba(255,255,255,0.24)', 4, focusStage, showFlowEdges, showSelfRoutes),
        edgeTrace(activePayload, 'route', '#f28482', 6, focusStage, showFlowEdges, showSelfRoutes),
        nodeTrace(activePayload, 'io', focusStage),
        nodeTrace(activePayload, 'dense', focusStage),
        nodeTrace(activePayload, 'module', focusStage),
      ];
    }}

    function routingHeatmapData(activePayload) {{
      if (!activePayload.routing_matrices.length) {{
        return [{{
          type: 'heatmap',
          z: [[1]],
          x: ['dense'],
          y: ['dense'],
          colorscale: [[0, '#1d2a44'], [1, '#90be6d']],
          showscale: false,
          hovertemplate: 'Dense model uses full block-to-block flow rather than per-stage routing masks<extra></extra>',
        }}];
      }}
      const traces = [];
      activePayload.routing_matrices.forEach((matrix, index) => {{
        traces.push({{
          type: 'heatmap',
          z: matrix.values,
          x: matrix.labels,
          y: matrix.labels,
          colorscale: [[0, '#1d2a44'], [1, '#f28482']],
          xaxis: `x${{index + 1}}`,
          yaxis: `y${{index + 1}}`,
          showscale: index === activePayload.routing_matrices.length - 1,
          colorbar: index === activePayload.routing_matrices.length - 1 ? {{ title: 'allowed' }} : undefined,
          hovertemplate: 'source=%{{x}}<br>destination=%{{y}}<br>allowed=%{{z}}<extra>' + matrix.label + '</extra>',
        }});
      }});
      return traces;
    }}

    function routingHeatmapLayout(activePayload) {{
      if (!activePayload.routing_matrices.length) {{
        return {{
          paper_bgcolor: '#0b1220',
          plot_bgcolor: '#0b1220',
          margin: {{ l: 80, r: 40, t: 50, b: 60 }},
          title: {{ text: 'Routing matrix', font: {{ color: '#e7edf7' }} }},
          xaxis: {{ color: '#9fb0c8' }},
          yaxis: {{ color: '#9fb0c8' }},
        }};
      }}
      const layout = {{
        paper_bgcolor: '#0b1220',
        plot_bgcolor: '#0b1220',
        margin: {{ l: 80, r: 40, t: 70, b: 50 }},
        title: {{ text: 'Routing adjacency by stage', font: {{ color: '#e7edf7' }} }},
        grid: {{ rows: activePayload.routing_matrices.length, columns: 1, pattern: 'independent' }},
        annotations: [],
      }};
      activePayload.routing_matrices.forEach((matrix, index) => {{
        layout.annotations.push({{
          text: matrix.label,
          x: 0.5,
          xref: 'paper',
          y: 1 - (index / Math.max(activePayload.routing_matrices.length, 1)) + 0.03,
          yref: 'paper',
          showarrow: false,
          font: {{ color: '#e7edf7', size: 14 }},
        }});
        layout[`xaxis${{index + 1}}`] = {{ color: '#9fb0c8', tickangle: -30 }};
        layout[`yaxis${{index + 1}}`] = {{ color: '#9fb0c8', autorange: 'reversed' }};
      }});
      return layout;
    }}

    function stageProfileTraces(activePayload) {{
      const labels = activePayload.stage_profiles.map((stage) => stage.label);
      return [
        {{
          type: 'bar',
          name: 'groups',
          x: labels,
          y: activePayload.stage_profiles.map((stage) => stage.group_count),
          marker: {{ color: '#84dcc6' }},
          hovertemplate: 'stage=%{{x}}<br>groups=%{{y}}<extra></extra>',
        }},
        {{
          type: 'bar',
          name: 'group width',
          x: labels,
          y: activePayload.stage_profiles.map((stage) => stage.group_dim),
          marker: {{ color: '#f6bd60' }},
          yaxis: 'y2',
          hovertemplate: 'stage=%{{x}}<br>group_dim=%{{y}}<extra></extra>',
        }},
        {{
          type: 'scatter',
          mode: 'lines+markers',
          name: 'local depth',
          x: labels,
          y: activePayload.stage_profiles.map((stage) => stage.depth),
          marker: {{ color: '#90be6d', size: 10 }},
          line: {{ color: '#90be6d', width: 3 }},
          hovertemplate: 'stage=%{{x}}<br>depth=%{{y}}<extra></extra>',
        }},
        {{
          type: 'scatter',
          mode: 'lines+markers',
          name: 'allowed routes',
          x: labels,
          y: activePayload.stage_profiles.map((stage) => stage.allowed_routes ?? 0),
          marker: {{ color: '#f28482', size: 10 }},
          line: {{ color: '#f28482', width: 3, dash: 'dot' }},
          hovertemplate: 'stage=%{{x}}<br>allowed_routes=%{{y}}<br>max_inbound=%{{customdata[0]}}<br>effective_cap=%{{customdata[1]}}<extra></extra>',
          customdata: activePayload.stage_profiles.map((stage) => [stage.max_inbound_routes ?? 0, stage.effective_inbound_cap ?? 0]),
        }},
      ];
    }}

    const stageProfileLayout = {{
      paper_bgcolor: '#0b1220',
      plot_bgcolor: '#0b1220',
      margin: {{ l: 60, r: 60, t: 70, b: 60 }},
      title: {{ text: 'Stage profile', font: {{ color: '#e7edf7' }} }},
      barmode: 'group',
      xaxis: {{ color: '#9fb0c8' }},
      yaxis: {{ title: 'groups / depth', color: '#9fb0c8', gridcolor: 'rgba(255,255,255,0.08)' }},
      yaxis2: {{ title: 'group width', overlaying: 'y', side: 'right', color: '#9fb0c8' }},
      legend: {{ font: {{ color: '#e7edf7' }} }},
    }};

    function refreshStageFocusOptions(activePayload) {{
      const selector = document.getElementById('stage-focus-select');
      const previousValue = selector.value || 'all';
      selector.innerHTML = '';
      selector.append(new Option('All stages', 'all'));
      activePayload.stage_profiles.forEach((stage, index) => selector.append(new Option(stage.label, String(index))));
      selector.value = Array.from(selector.options).some((option) => option.value === previousValue) ? previousValue : 'all';
    }}

    function syncControlsFromPayload(activePayload) {{
      const config = activePayload.model_config;
      const usesColumns = config.architecture === 'cortical_columns';
      document.getElementById('stage-groups-input').value = (usesColumns ? (config.column_counts || []) : (config.stage_groups || [])).join(',');
      document.getElementById('stage-depths-input').value = (usesColumns ? (config.column_depths || []) : (config.stage_depths || [])).join(',');
      document.getElementById('fixed-group-size-input').value = (() => {{
        const value = usesColumns ? config.fixed_column_size : config.fixed_group_size;
        return value == null ? '' : String(value);
      }})();
      document.getElementById('routing-topology-select').value = usesColumns ? (config.column_routing_topology || config.routing_topology) : config.routing_topology;
      document.getElementById('routing-top-k-input').value = (() => {{
        const value = usesColumns ? (config.column_routing_top_k == null ? config.routing_top_k : config.column_routing_top_k) : config.routing_top_k;
        return value == null ? '' : String(value);
      }})();
      document.getElementById('model-config').textContent = JSON.stringify(config, null, 2);
    }}

    function renderAll(activePayload) {{
      refreshStageFocusOptions(activePayload);
      syncControlsFromPayload(activePayload);
      Plotly.react('structure-plot', structureTraces(activePayload), structureLayout, {{ responsive: true, displaylogo: false }});
      Plotly.react('routing-plot', routingHeatmapData(activePayload), routingHeatmapLayout(activePayload), {{ responsive: true, displaylogo: false }});
      Plotly.react('profile-plot', stageProfileTraces(activePayload), stageProfileLayout, {{ responsive: true, displaylogo: false }});
    }}

    renderAll(payload);

    function activateView(targetId) {{
      document.querySelectorAll('.view-button').forEach((button) => button.classList.toggle('active', button.dataset.target === targetId));
      document.querySelectorAll('.plot-surface').forEach((surface) => surface.classList.toggle('active', surface.id === targetId));
      window.dispatchEvent(new Event('resize'));
    }}

    document.querySelectorAll('.view-button').forEach((button) => {{
      button.addEventListener('click', () => activateView(button.dataset.target));
    }});

    document.getElementById('show-flow-toggle').addEventListener('change', () => Plotly.react('structure-plot', structureTraces(payload), structureLayout, {{ responsive: true, displaylogo: false }}));
    document.getElementById('show-self-routes-toggle').addEventListener('change', () => Plotly.react('structure-plot', structureTraces(payload), structureLayout, {{ responsive: true, displaylogo: false }}));
    document.getElementById('stage-focus-select').addEventListener('change', () => Plotly.react('structure-plot', structureTraces(payload), structureLayout, {{ responsive: true, displaylogo: false }}));
    document.getElementById('apply-preview').addEventListener('click', () => {{
      try {{
        const nextModelConfig = cloneValue(payload.model_config);
        const usesColumns = nextModelConfig.architecture === 'cortical_columns';
        const originalGroups = usesColumns ? (payload.model_config.column_counts || [2, 1]) : (payload.model_config.stage_groups || [2, 1]);
        const nextGroups = parseIntegerList(document.getElementById('stage-groups-input').value, originalGroups);
        const nextDepths = parseIntegerList(document.getElementById('stage-depths-input').value, (usesColumns ? payload.model_config.column_depths : payload.model_config.stage_depths) || Array.from({{ length: nextGroups.length }}, () => 1));
        const fixedGroupSizeRaw = document.getElementById('fixed-group-size-input').value.trim();
        const nextFixedGroupSize = fixedGroupSizeRaw ? Number.parseInt(fixedGroupSizeRaw, 10) : null;
        if (nextFixedGroupSize != null && (!Number.isInteger(nextFixedGroupSize) || nextFixedGroupSize <= 0)) {{
          throw new Error('fixed_group_size must be a positive integer when provided.');
        }}
        const topKRaw = document.getElementById('routing-top-k-input').value.trim();
        const nextRoutingTopK = topKRaw ? Number.parseInt(topKRaw, 10) : null;
        if (nextRoutingTopK != null && (!Number.isInteger(nextRoutingTopK) || nextRoutingTopK <= 0)) {{
          throw new Error('routing_top_k must be a positive integer when provided.');
        }}
        if (usesColumns) {{
          nextModelConfig.column_counts = nextGroups;
          nextModelConfig.column_depths = nextDepths;
          nextModelConfig.fixed_column_size = nextFixedGroupSize;
          nextModelConfig.column_routing_topology = document.getElementById('routing-topology-select').value;
          nextModelConfig.column_routing_top_k = nextRoutingTopK;
        }} else {{
          nextModelConfig.stage_groups = nextGroups;
          nextModelConfig.stage_depths = nextDepths;
          nextModelConfig.fixed_group_size = nextFixedGroupSize;
          nextModelConfig.routing_topology = document.getElementById('routing-topology-select').value;
          nextModelConfig.routing_top_k = nextRoutingTopK;
        }}
        payload = buildPayloadFromModelConfig(nextModelConfig);
        renderAll(payload);
      }} catch (error) {{
        window.alert(error.message);
      }}
    }});
    document.getElementById('reset-preview').addEventListener('click', () => {{
      payload = cloneValue(originalPayload);
      document.getElementById('show-flow-toggle').checked = true;
      document.getElementById('show-self-routes-toggle').checked = true;
      renderAll(payload);
    }});
  </script>
</body>
</html>
"""


def write_structure_html(config: PrometheusConfig, output_path: str | Path) -> Path:
    """Write the structure visualization HTML to disk."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_structure_html(config), encoding="utf-8")
    return destination


def _add_dense_structure(payload: dict[str, Any], embedding_dim: int, num_layers: int) -> None:
    payload["stages"] = [{"label": f"Dense block {index + 1}"} for index in range(num_layers)]
    payload["stage_profiles"] = [
        {
            "label": f"Dense {index + 1}",
            "group_count": 1,
            "group_dim": embedding_dim,
            "depth": 1,
      "allowed_routes": 1,
      "max_inbound_routes": 1,
      "effective_inbound_cap": 1,
        }
        for index in range(num_layers)
    ]
    previous_id = "input"
    for index in range(num_layers):
        node_id = f"dense-{index}"
        payload["nodes"].append(
            {
                "id": node_id,
                "label": f"Dense {index + 1}",
                "kind": "dense",
                "stage_index": index,
                "group_index": 0,
                "x": float(index + 1),
                "y": 0.0,
                "z": 0.0,
                "details": f"embedding_dim={embedding_dim}",
            }
        )
        _append_edge(payload, previous_id, node_id, kind="flow")
        previous_id = node_id


def _add_modular_structure(payload: dict[str, Any], stage_specs: list[Any], routing_topology: str, routing_top_k: int | None) -> None:
    payload["stages"] = []
    previous_stage_nodes = ["input"]
    for stage_index, stage_spec in enumerate(stage_specs):
        group_count = stage_spec.group_count
        group_dim = stage_spec.group_dim
        depth = stage_spec.depth
        routing_mask = StaticRouter._build_mask(group_count, routing_topology)
        max_inbound_routes = int(routing_mask.sum(dim=1).max().item())
        payload["stages"].append({"label": f"Stage {stage_index + 1}", "group_count": group_count, "depth": depth})
        payload["stage_profiles"].append(
            {
                "label": f"Stage {stage_index + 1}",
                "group_count": group_count,
                "group_dim": group_dim,
                "stage_dim": stage_spec.stage_dim,
                "depth": depth,
                "allowed_routes": int(routing_mask.sum().item()),
                "max_inbound_routes": max_inbound_routes,
                "effective_inbound_cap": min(max_inbound_routes, routing_top_k) if routing_top_k is not None else max_inbound_routes,
            }
        )
        current_stage_nodes = []
        y_center = (group_count - 1) / 2.0
        for group_index in range(group_count):
            node_id = f"stage-{stage_index}-group-{group_index}"
            payload["nodes"].append(
                {
                    "id": node_id,
                    "label": f"S{stage_index + 1}G{group_index + 1}",
                    "kind": "module",
                    "stage_index": stage_index,
                    "group_index": group_index,
                    "x": float(stage_index + 1),
                    "y": float(group_index - y_center),
                    "z": 0.0,
                    "details": f"stage_dim={stage_spec.stage_dim}, group_dim={group_dim}, depth={depth}",
                }
            )
            current_stage_nodes.append(node_id)
            for previous_id in previous_stage_nodes:
                _append_edge(payload, previous_id, node_id, kind="flow")
        payload["routing_matrices"].append(
            {
                "label": f"Stage {stage_index + 1}",
                "labels": [f"G{group_index + 1}" for group_index in range(group_count)],
                "values": routing_mask.to(dtype=torch.int).tolist(),
            }
        )
        for destination in range(group_count):
            for source in range(group_count):
                if not bool(routing_mask[destination, source]):
                    continue
                _append_edge(
                    payload,
                    f"stage-{stage_index}-group-{source}",
                    f"stage-{stage_index}-group-{destination}",
                    kind="route",
                    z_offset=0.9,
                    details=f"topology={routing_topology}, top_k={routing_top_k}",
                )
        previous_stage_nodes = current_stage_nodes


def _append_edge(payload: dict[str, Any], source_id: str, target_id: str, kind: str, z_offset: float = 0.0, details: str = "") -> None:
    source = _node_by_id(payload, source_id)
    target = _node_by_id(payload, target_id)
    payload["edges"].append(
        {
            "source": source_id,
            "target": target_id,
            "kind": kind,
            "details": details,
            "x0": source["x"],
            "y0": source["y"],
            "z0": source["z"] + z_offset,
            "x1": target["x"],
            "y1": target["y"],
            "z1": target["z"] + z_offset,
        }
    )


def _node_by_id(payload: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in payload["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"Unknown node id: {node_id}")


def _connect_last_stage_to_output(payload: dict[str, Any]) -> None:
    output_id = "output"
    if payload["architecture"] == "dense":
        if payload["model_config"]["num_layers"] == 0:
            _append_edge(payload, "input", output_id, kind="flow")
            return
        _append_edge(payload, f"dense-{payload['model_config']['num_layers'] - 1}", output_id, kind="flow")
        return
    final_stage_index = len(payload["stage_profiles"]) - 1
    final_group_count = payload["stage_profiles"][-1]["group_count"]
    for group_index in range(final_group_count):
        _append_edge(payload, f"stage-{final_stage_index}-group-{group_index}", output_id, kind="flow")