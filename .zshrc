# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
fastfetch
alias install="sudo apt install"
alias update="sudo apt update"
alias upgrade="sudo apt upgrade"
alias remove="sudo apt remove --purge"
alias restart="sudo reboot now"
alias shutdown="sudo shutdown now"
alias ..="cd .."
export PATH="$PATH:/home/killian/.local/bin"
eval "$(oh-my-posh init zsh --config ~/.cache/oh-my-posh/themes/agnosterplus.omp.json)"
