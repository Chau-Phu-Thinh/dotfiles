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
alias python="python3"
alias ..="cd .."

export STARSHIP_CONFIG=$HOME/.config/starship/starship.toml
eval "$(starship init zsh)"
# opencode
export PATH=/home/killua/.opencode/bin:$PATH

# Added by Antigravity CLI installer
export PATH="/home/killua/.local/bin:$PATH"
