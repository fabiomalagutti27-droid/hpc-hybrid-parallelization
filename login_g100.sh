#!/bin/bash

# --- CONFIGURAZIONE ---
# VERIFICA QUESTO USERNAME DALL'EMAIL CINECA!
USERNAME="fmalagut" 

# La tua email istituzionale
EMAIL="fabio.malagutti2@studio.unibo.it"

# Il cluster Galileo100
HOST="login.g100.cineca.it"

# Parametri di autenticazione CINECA
PROVISIONER="cineca-hpc"

# CORREZIONE: Sul tuo Mac il comando è 'step', non 'step-cli'
STEP_COMMAND="step" 

echo "--- Avvio connessione CINECA HPC (Galileo100) ---"

# Imposta il terminale corretto per evitare errori grafici
TERM=xterm
export TERM

echo "1. Avvio SSH Agent temporaneo..."
eval "$(ssh-agent -s)"
AGENT_PID=$!
if [ $? -ne 0 ]; then
  echo "ERRORE: Impossibile avviare SSH agent."
  exit 1
fi

echo "2. Autenticazione OIDC..."
# Questo comando aprirà il browser per il login Unibo
"${STEP_COMMAND}" ssh login "${EMAIL}" --provisioner "${PROVISIONER}"

if [ $? -ne 0 ]; then
  echo "ERRORE: Login fallito. Assicurati di aver completato l'accesso nel browser."
  kill "${AGENT_PID}" >/dev/null 2>&1
  exit 1
fi

echo "3. Pulizia vecchie chiavi (per evitare conflitti)..."
ssh-keygen -R "${HOST}" 2>/dev/null

echo "4. Connessione a ${USERNAME}@${HOST}..."
# Si connette passando la variabile TERM
ssh -o SendEnv=TERM "${USERNAME}@${HOST}"

# Quando esci dal cluster (logout), lo script continua qui:
echo "Chiusura connessione e pulizia SSH agent..."
ssh-agent -k >/dev/null 2>&1
echo "Disconnesso. A presto!"
