import { invoke } from "@tauri-apps/api/core";

let inputElement: HTMLInputElement | null;
let outputElement: HTMLElement | null;

async function punctuate() {
  console.log('punctuate clicked')
  if (outputElement && inputElement) {
    // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
    try {
      const result = await invoke<string>("punctuate", {
        sentence: inputElement.value,
      });
      console.log('result', result)
      outputElement.textContent = result
    } catch (error) {
      console.error(error)
    }
    
  }
}

window.addEventListener("DOMContentLoaded", () => {
  inputElement = document.querySelector("#input");
  outputElement = document.querySelector("#output");
  document.querySelector("#input-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    punctuate();
  });
});
