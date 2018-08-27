from itertools import cycle
import random
import sys
import numpy as np

import pygame
from pygame.locals import *

class FlappyGame:
    FPS = 60 
    SCREENWIDTH  = 200
    SCREENHEIGHT = 500
    BIRDWIDTH = 35
    BIRDHEIGHT = 25
    PIPEWIDTH = 50
    PIPEHEIGHT = 320
    # amount by which base can maximum shift to left
    PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
    BASEY        = SCREENHEIGHT - 100
    # image, sound and hitmask  dicts
    IMAGES = {}


    try:
        xrange
    except NameError:
        xrange = range

    
    def init_world(self):
        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # select random background sprites
        self.IMAGES['background'] = pygame.image.load('assets/sprites/background-day.png' ).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load('assets/sprites/redbird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/redbird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/redbird-downflap.png').convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(), 180),
            pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(),
        )
    
    def learn(self, bot, num_episodes, learn = True, show_screen = False, discretization = 5, seed = None, evaluation = False):
        self.discretization = discretization
        self.learning = learn
        self.show_screen = show_screen
        self.evaluation = evaluation
        self.eval_scores = []
    
        if self.show_screen:
            pygame.init()
            self.FPSCLOCK = pygame.time.Clock()
            self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
            pygame.display.set_caption('Flappy Bird')

            self.init_world()
            
        cnt = 0
        while cnt < num_episodes:
            if seed is None:
                random.seed(cnt)
                #print("seed:", cnt)
            else:
                random.seed(seed)
                #print("seed:", seed)
            
            cnt += 1
            crashInfo = self.mainGame(bot)

        if self.evaluation:
            print("evaluation")
            print(np.mean(self.eval_scores), np.std(self.eval_scores),
                             np.min(self.eval_scores), np.max(self.eval_scores))
    

    def getDemos(self):
        self.learning = False
        self.show_screen = True
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        self.init_world()
       
        while True:
           

            self.showWelcomeAnimation()
            crashInfo = self.mainGame()
            self.showGameOverScreen(crashInfo)


    def showWelcomeAnimation(self):
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        playerIndex = 0
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        loopIter = 0

        playerx = self.SCREENWIDTH // 5 
        playery = self.SCREENHEIGHT // 2

        messagex = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        messagey = int(self.SCREENHEIGHT * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()



        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    # make first flap sound and return values for mainGame
                      return {
                        'playery': playery,  
                        'basex': basex,
                    }


            basex = -((-basex + 5) % baseShift)


            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))
            self.SCREEN.blit(self.IMAGES['player'][playerIndex],
                        (playerx, playery))
            self.SCREEN.blit(self.IMAGES['message'], (messagex, messagey))
            self.SCREEN.blit(self.IMAGES['base'], (basex, self.BASEY))

            pygame.display.update()
            self.FPSCLOCK.tick(self.FPS)


    def mainGame(self, bot = None):
        score = playerIndex = loopIter = 0
        playerx = self.SCREENWIDTH // 5 
        playery = self.SCREENHEIGHT // 2
        if self.show_screen:
            basex =0
            baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        # list of upper pipes
        upperPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + self.SCREENWIDTH , 'y': newPipe2[0]['y']},
        ]

        # list of lowerpipe
        lowerPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + self.SCREENWIDTH, 'y': newPipe2[1]['y']},
        ]

        pipeVelX = -5

        # player velocity, max velocity, downward accleration, accleration on flap
        playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        playerMaxVelY =  10   # max vel along Y, max descend speed
        playerMinVelY =  -8   # min vel along Y, max ascend speed
        playerAccY    =   1   # players downward accleration
        playerRot     =  45   # player's rotation
        playerVelRot  =   3   # angular speed
        playerRotThr  =  20   # rotation threshold
        playerFlapAcc =  -9   # players speed on flapping
        playerFlapped = False # True when player flaps

        activePipe = 0
        traj = []

        while True:
        
            if bot is not None:
                s = (round((playery-lowerPipes[activePipe]['y'])/self.discretization), round((lowerPipes[activePipe]['x']-playerx)/self.discretization), playerVelY)
                #s = (round((playery-lowerPipes[activePipe]['y'])/discretisation), playerVelY)
                bot.appendState(s)
                max_act, max_val = bot.maxQ(s)

        
        
                if max_act:
                    if playery > 0:
                        playerVelY = playerFlapAcc
                        playerFlapped = True
        
        
        
            if self.show_screen:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        bot.saveQ()
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                        if playery > -2 * self.IMAGES['player'][0].get_height():
                            playerVelY = playerFlapAcc
                            playerFlapped = True


          

            if self.show_screen:
                basex = -((-basex + 100) % baseShift)


            # player's movement
            if playerVelY < playerMaxVelY and not playerFlapped:
                playerVelY += playerAccY
            if playerFlapped:
                playerFlapped = False



            playerHeight = self.BIRDHEIGHT
            playery += min(playerVelY, self.BASEY - playery - playerHeight)

            # move pipes to left
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipe['x'] += pipeVelX
                lPipe['x'] += pipeVelX

            # add new pipe when first pipe is about to touch left of screen
            if 0 == upperPipes[0]['x']:
                newPipe = self.getRandomPipe()
                upperPipes.append(newPipe[0])
                lowerPipes.append(newPipe[1])

            # remove first pipe if its out of the screen
            if upperPipes[0]['x'] < -self.PIPEWIDTH:
                upperPipes.pop(0)
                lowerPipes.pop(0)
                activePipe-=1
                
                
            # check for crash here
            crashTest = self.checkCrash({'x': playerx, 'y': playery},
                                   upperPipes, lowerPipes)
            

            # check for score
            scored = False
            playerPos = playerx
            for pipe in upperPipes:
                pipeEndPos = pipe['x'] + self.PIPEWIDTH
                if pipeEndPos <= playerPos < pipeEndPos + 5:
                    score += 1
                    activePipe+=1
                    scored = True
                    
            if self.learning and bot is not None:
                if scored:
                    r = 1
                elif crashTest:
                    r = -10
                else:
                    r = 0
            
            if self.learning and bot is not None:
                # Create experience tuple
                s2 = (round((playery-lowerPipes[activePipe]['y'])/self.discretization), round((lowerPipes[activePipe]['x']-playerx)/self.discretization), playerVelY)
                #s2 = (round((playery-lowerPipes[activePipe]['y'])/discretisation), playerVelY)
                bot.appendState(s2)
                traj.append((s, max_act, r, s2))
                
            if crashTest:
                if self.evaluation:
                    self.eval_scores.append(score)
                if score == 4:
                    pass
                    #input("pause")
                if self.learning and bot is not None:
                    bot.updateQ(traj, score)
                return {
                    'y': playery,
                    'upperPipes': upperPipes,
                    'lowerPipes': lowerPipes,
                    'score': score,
                    'playerVelY': playerVelY,
                    'playerRot': playerRot
                }
            elif score >= 100:
                if self.evaluation:
                    self.eval_scores.append(score)
                if self.learning and bot is not None:
                    bot.updateQ(traj, score)
                return {
                    'y': playery,
                    'upperPipes': upperPipes,
                    'lowerPipes': lowerPipes,
                    'score': score,
                    'playerVelY': playerVelY,
                    'playerRot': playerRot
                }  
                

            # draw sprites
            if self.show_screen:
                self.SCREEN.blit(self.IMAGES['background'], (0,0))

                for uPipe, lPipe in zip(upperPipes, lowerPipes):
                    self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                    self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

                self.SCREEN.blit(self.IMAGES['base'], (basex, self.BASEY))
                # print score so player overlaps the score
                self.showScore(score)


                
                playerSurface = self.IMAGES['player'][playerIndex]
                self.SCREEN.blit(playerSurface, (playerx, playery))

                pygame.display.update()
                self.FPSCLOCK.tick(self.FPS)


    def showGameOverScreen(self, crashInfo):
        """crashes the player down ans shows gameover image"""
        score = crashInfo['score']
        playerx = self.SCREENWIDTH * 0.2
        playery = crashInfo['y']
        playerHeight = self.IMAGES['player'][0].get_height()
        playerVelY = crashInfo['playerVelY']
        playerAccY = 2
        playerRot = crashInfo['playerRot']
        playerVelRot = 7

        basex = 0

        upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']



        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    if playery + playerHeight >= self.BASEY - 1:
                        return

            # player y shift
            if playery + playerHeight < self.BASEY - 1:
                playery += min(playerVelY, self.BASEY - playery - playerHeight)

            # player velocity change
            if playerVelY < 15:
                playerVelY += playerAccY



            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.SCREEN.blit(self.IMAGES['base'], (basex, self.BASEY))
            self.showScore(score)

            playerSurface = pygame.transform.rotate(self.IMAGES['player'][1], playerRot)
            self.SCREEN.blit(playerSurface, (playerx,playery))

            self.FPSCLOCK.tick(self.FPS)
            pygame.display.update()




    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.choice([100, 150, 200])
        
        pipeX = self.SCREENWIDTH + self.SCREENWIDTH

        return [
            {'x': pipeX, 'y': gapY - self.PIPEHEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE}, # lower pipe
        ]


    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()


    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes or ceiling."""


        # if player crashes into ground
        if player['y'] + self.BIRDHEIGHT >= self.BASEY:
            return True
        elif player['y'] <= 0:
            return True
        else:
            bird_l = (player['x'], player['y'])
            #print((player['x'], player['y']))
            #print(BIRDWIDTH, BIRDHEIGHT)
            bird_r = (player['x'] + self.BIRDWIDTH, player['y'] + self.BIRDHEIGHT)
            #print("bird:", bird_l, bird_r)
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe upper left points
                lPipe_l = (lPipe['x'], lPipe['y'])
                uPipe_l = (lPipe['x'], 0)
                

                lPipe_r = (lPipe['x'] + self.PIPEWIDTH, self.SCREENHEIGHT)            
                uPipe_r = (lPipe['x'] + self.PIPEWIDTH, lPipe['y'] - self.PIPEGAPSIZE)

                #print("upperpipe:", uPipe_l, uPipe_r)
                #print("lowerpipe:", lPipe_l, lPipe_r)
                

                # if bird collided with upipe or lpipe
                uCollide = self.inCollision(uPipe_l, uPipe_r, bird_l, bird_r)
                #print("upper collision", uCollide)
                
                lCollide = self.inCollision(lPipe_l, lPipe_r, bird_l, bird_r)
                #print("lower collision", lCollide)
                if uCollide or lCollide:
                    return True

            return False

    def inCollision(self, l1, r1, l2, r2):
        
        # If one rectangle is on left side of other
        if l1[0] > r2[0] or l2[0] > r1[0]:
            return False
        # If one rectangle is above other
        if l1[1] > r2[1] or l2[1] > r1[1]:
            return False
     
        return True



if __name__ == '__main__':
    game = FlappyGame()
    game.getDemos() 
