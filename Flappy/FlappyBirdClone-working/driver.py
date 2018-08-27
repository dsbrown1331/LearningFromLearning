import flappy
import bot as agent
bot = agent.Bot()
game = flappy.FlappyGame()
#game.getDemos()

#works!
for i in range(10):
    bot.set_eps(0.001)
    num_episodes = 1000
    game.learn(bot, num_episodes, True, False, discretization = 3)
num_eval = 1000
bot.set_eps(0)
game.learn(bot, num_eval, False, False, discretization = 3, evaluation = True)
