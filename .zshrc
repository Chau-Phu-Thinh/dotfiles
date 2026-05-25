# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
fastfetch
alias install="sudo apt-get install"
alias update="sudo apt update"
alias upgrade="sudo apt upgrade"
alias remove="sudo apt remove --purge"
alias restart="sudo reboot now"
alias shutdown="sudo shutdown now"
alias python="python3"
alias ..="cd .."
export PATH="$PATH:/home/killua/.local/bin"

COLOR_MODE=$(gsettings get org.gnome.desktop.interface color-scheme)

if [[ "$COLOR_MODE" == *"default"* ]]; then
    POSH_THEME="$HOME/dotfiles/.config/oh-my-posh/themes/light-clean-detailed.omp.json"
else
    POSH_THEME="$HOME/dotfiles/.config/oh-my-posh/themes/dark-clean-detailed.omp.json"
fi

eval "$(oh-my-posh init zsh --config "$POSH_THEME")"

# opencode
export PATH=/home/killua/.opencode/bin:$PATH


# Added by Antigravity CLI installer
export PATH="/home/killua/.local/bin:$PATH"
