# IAG Assignment

**Intelligent Agents (INFOIAG)**  
Utrecht University

This repository contains the assignment for the Intelligent Agents course at Utrecht University.

## Prerequisites

- [Ollama](https://ollama.com/)
- Windows with PowerShell

## Getting Started

### 1. Install Ollama

Open PowerShell and run:

```powershell
irm https://ollama.com/install.ps1 | iex
```

### 2. Download the Language Model

Download the Llama 3.1 8B model:

```powershell
ollama pull llama3.1:8b
```

### 3. Verify the Installation

Check that Ollama is installed correctly:

```powershell
ollama --version
```

Verify that the model is available:

```powershell
ollama list
```