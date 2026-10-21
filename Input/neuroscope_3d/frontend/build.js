/**
 * Frontend build script for NeuroScope 3D.
 * Transpiles TypeScript & React TSX using @babel/core and packages into frontend/dist.
 */

const fs = require('fs');
const path = require('path');
const babel = require('/usr/share/nodejs/@babel/core');

const FRONTEND_DIR = __dirname;
const SRC_DIR = path.join(FRONTEND_DIR, 'src');
const DIST_DIR = path.join(FRONTEND_DIR, 'dist');

if (!fs.existsSync(DIST_DIR)) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

const filesToCompile = [
  { key: './types', path: path.join(SRC_DIR, 'types', 'index.ts'), isTypeOnly: true },
  { key: './utils/modalityDetector', path: path.join(SRC_DIR, 'utils', 'modalityDetector.ts') },
  { key: './components/MedicalDisclaimer', path: path.join(SRC_DIR, 'components', 'MedicalDisclaimer.tsx') },
  { key: './components/UploadSlot', path: path.join(SRC_DIR, 'components', 'UploadSlot.tsx') },
  { key: './components/ValidationSummary', path: path.join(SRC_DIR, 'components', 'ValidationSummary.tsx') },
  { key: './components/SegmentationResult', path: path.join(SRC_DIR, 'components', 'SegmentationResult.tsx') },
  { key: './App', path: path.join(SRC_DIR, 'App.tsx') },
];

let compiledModules = {};

for (const item of filesToCompile) {
  if (item.isTypeOnly) continue;
  const sourceCode = fs.readFileSync(item.path, 'utf8');
  const result = babel.transformSync(sourceCode, {
    filename: item.path,
    presets: [
      '/usr/share/nodejs/@babel/preset-typescript',
      '/usr/share/nodejs/@babel/preset-react',
    ],
    plugins: [
      '/usr/share/nodejs/@babel/plugin-transform-modules-commonjs',
    ],
  });
  compiledModules[item.key] = result.code;
}

const cssContent = fs.readFileSync(path.join(SRC_DIR, 'index.css'), 'utf8');

let bundleJs = `
(function() {
  var modules = {};
  function define(name, fn) { modules[name] = { fn: fn }; }
  function requireModule(name) {
    if (name === 'react') return window.React;
    if (name === 'react-dom') return window.ReactDOM;
    if (name === './types' || name.indexOf('/types') !== -1) return {};

    var key = name;
    if (!modules[key]) {
      var candidates = [
        key,
        key + '.tsx',
        key + '.ts',
        key + '.js',
        './' + key,
        key.replace(/^\\.\\//, '')
      ];
      for (var i = 0; i < candidates.length; i++) {
        if (modules[candidates[i]]) {
          key = candidates[i];
          break;
        }
      }
    }

    if (!modules[key]) {
      var keys = Object.keys(modules);
      for (var j = 0; j < keys.length; j++) {
        if (keys[j].indexOf(name.replace(/^\\.\\//, '')) !== -1) {
          key = keys[j];
          break;
        }
      }
    }

    if (!modules[key]) {
      throw new Error('Module not found: ' + name);
    }

    if (!modules[key].cached) {
      var mod = { exports: {} };
      modules[key].fn(mod, mod.exports, requireModule);
      modules[key].cached = mod.exports;
    }
    return modules[key].cached;
  }
`;

for (const [key, code] of Object.entries(compiledModules)) {
  bundleJs += `
  define(${JSON.stringify(key)}, function(module, exports, require) {
${code}
  });
`;
}

bundleJs += `
  window.addEventListener('DOMContentLoaded', function() {
    var App = requireModule('./App').App;
    var container = document.getElementById('root');
    if (window.ReactDOM && window.ReactDOM.createRoot) {
      var root = window.ReactDOM.createRoot(container);
      root.render(window.React.createElement(App, null));
    } else if (window.ReactDOM && window.ReactDOM.render) {
      window.ReactDOM.render(window.React.createElement(App, null), container);
    }
  });
})();
`;

const offlineReactEngine = `
<script>
(function() {
  if (window.React && window.ReactDOM) return;

  var currentHookState = {
    currentFiber: null,
    hookIndex: 0
  };

  function useState(initialValue) {
    var fiber = currentHookState.currentFiber;
    var index = currentHookState.hookIndex++;
    if (!fiber.hooks) fiber.hooks = [];
    if (fiber.hooks.length <= index) {
      fiber.hooks[index] = typeof initialValue === 'function' ? initialValue() : initialValue;
    }
    var setState = function(newValue) {
      var next = typeof newValue === 'function' ? newValue(fiber.hooks[index]) : newValue;
      if (fiber.hooks[index] !== next) {
        fiber.hooks[index] = next;
        scheduleRender();
      }
    };
    return [fiber.hooks[index], setState];
  }

  function useRef(initialValue) {
    var fiber = currentHookState.currentFiber;
    var index = currentHookState.hookIndex++;
    if (!fiber.hooks) fiber.hooks = [];
    if (fiber.hooks.length <= index) {
      fiber.hooks[index] = { current: initialValue };
    }
    return fiber.hooks[index];
  }

  function createElement(type, props) {
    var children = [];
    for (var i = 2; i < arguments.length; i++) {
      var arg = arguments[i];
      if (Array.isArray(arg)) {
        for (var j = 0; j < arg.length; j++) {
          if (arg[j] != null && arg[j] !== false && arg[j] !== true) children.push(arg[j]);
        }
      } else if (arg != null && arg !== false && arg !== true) {
        children.push(arg);
      }
    }
    props = Object.assign({}, props);
    return { type: type, props: props, children: children };
  }

  var Fragment = function(props) { return props.children; };

  var rootContainer = null;
  var rootElement = null;
  var appFiber = { hooks: [] };
  var renderPending = false;

  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(function() {
      renderPending = false;
      renderApp();
    });
  }

  function renderApp() {
    if (!rootContainer || !rootElement) return;
    currentHookState.currentFiber = appFiber;
    currentHookState.hookIndex = 0;

    var vdom = null;
    if (typeof rootElement.type === 'function') {
      vdom = rootElement.type(Object.assign({}, rootElement.props, { children: rootElement.children }));
    } else {
      vdom = rootElement;
    }

    var dom = createDOM(vdom);
    rootContainer.innerHTML = '';
    if (dom) rootContainer.appendChild(dom);
  }

  function createDOM(vnode) {
    if (vnode == null || vnode === false || vnode === true) return null;
    if (typeof vnode === 'string' || typeof vnode === 'number') {
      return document.createTextNode(String(vnode));
    }
    if (Array.isArray(vnode)) {
      var frag = document.createDocumentFragment();
      for (var i = 0; i < vnode.length; i++) {
        var child = createDOM(vnode[i]);
        if (child) frag.appendChild(child);
      }
      return frag;
    }

    if (typeof vnode.type === 'function') {
      var res = vnode.type(Object.assign({}, vnode.props, { children: vnode.children }));
      return createDOM(res);
    }

    var el = document.createElement(vnode.type);
    var props = vnode.props || {};

    for (var key in props) {
      if (key === 'children' || key === 'key') continue;
      if (key === 'ref' && props[key]) {
        props[key].current = el;
      } else if (key.startsWith('on') && typeof props[key] === 'function') {
        var eventName = key.substring(2).toLowerCase();
        el.addEventListener(eventName, props[key]);
      } else if (key === 'className') {
        el.className = props[key];
      } else if (key === 'disabled') {
        el.disabled = Boolean(props[key]);
      } else if (key === 'value') {
        el.value = props[key];
      } else if (key === 'style' && typeof props[key] === 'object') {
        Object.assign(el.style, props[key]);
      } else {
        el.setAttribute(key, props[key]);
      }
    }

    var children = vnode.children || [];
    for (var j = 0; j < children.length; j++) {
      var cNode = createDOM(children[j]);
      if (cNode) el.appendChild(cNode);
    }

    return el;
  }

  window.React = {
    createElement: createElement,
    Fragment: Fragment,
    useState: useState,
    useRef: useRef
  };

  window.ReactDOM = {
    createRoot: function(container) {
      rootContainer = container;
      return {
        render: function(element) {
          rootElement = element;
          renderApp();
        }
      };
    },
    render: function(element, container) {
      rootContainer = container;
      rootElement = element;
      renderApp();
    }
  };
})();
</script>
`;

const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NeuroScope 3D — NIfTI Upload & Validation Module</title>
  <style>
${cssContent}
  </style>
</head>
<body>
  <div id="root"></div>

  <!-- CDN scripts if online -->
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>

  <!-- Fallback full React engine if offline -->
  ${offlineReactEngine}

  <!-- App Bundle -->
  <script>
${bundleJs}
  </script>
</body>
</html>
`;

fs.writeFileSync(path.join(DIST_DIR, 'index.html'), htmlContent);
fs.writeFileSync(path.join(FRONTEND_DIR, 'index.html'), htmlContent);
console.log('Frontend built successfully: ' + path.join(DIST_DIR, 'index.html'));
