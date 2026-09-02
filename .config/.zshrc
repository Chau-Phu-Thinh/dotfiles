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
alias lsd="tree -d"
export STARSHIP_CONFIG=$HOME/.config/starship/starship.toml
eval "$(starship init zsh)"
# opencode
#
export PATH=/home/killua/.opencode/bin:$PATH


# Added by Antigravity CLI installer
export PATH="/home/killua/.local/bin:$PATH"



export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
