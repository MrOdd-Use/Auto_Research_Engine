# npm package

The [Auto_Research_Engine npm package](https://www.npmjs.com/package/Auto_Research_Engine) is a WebSocket client for interacting with GPT Researcher.

## Installation

```bash
npm install Auto_Research_Engine
```

## Usage

```javascript
const GPTResearcher = require('Auto_Research_Engine');

const researcher = new GPTResearcher({
  host: 'localhost:8000',
  logListener: (data) => console.log('logListener logging data: ',data)
});

researcher.sendMessage({
  query: 'Does providing better context reduce LLM hallucinations?',
  moreContext: 'Provide a detailed answer'
});
```
