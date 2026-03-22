class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }
    this.port.postMessage(input[0].slice());
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
