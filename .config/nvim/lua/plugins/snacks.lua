return
-- lazy.nvim
{
  "folke/snacks.nvim",
  ---@type snacks.Config
  opts = {
    dashboard = {
      -- your dashboard configuration comes here
      -- or leave it empty to use the default settings
      -- refer to the configuration section below
      preset = {

        --        header = [[
        --    ███╗   ██╗███████╗ ██████╗ ██╗   ██╗██╗███╗   ███╗
        --    ████╗  ██║██╔════╝██╔═══██╗██║   ██║██║████╗ ████║
        --    ██╔██╗ ██║█████╗  ██║   ██║██║   ██║██║██╔████╔██║
        --    ██║╚██╗██║██╔══╝  ██║   ██║╚██╗ ██╔╝██║██║╚██╔╝██║
        --    ██║ ╚████║███████╗╚██████╔╝ ╚████╔╝ ██║██║ ╚═╝ ██║
        --    ╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═══╝  ╚═╝╚═╝     ╚═╝]],
        header = [[
█    ██   ▄▄▄▄▄▄ ▀▄    ▄ 
█    █ █ ▀   ▄▄▀   █  █  
█    █▄▄█ ▄▀▀   ▄▀  ▀█   
███▄ █  █ ▀▀▀▀▀▀    █    
    ▀   █         ▄▀     
       █                 
      ▀                  
        ]],
      },
      sections = {
        { section = "header", padding = 0 },

        { section = "keys", gap = 1, padding = 1 },
        { section = "startup" },
      },
    },
  },
}
