prompt_guandan = '''You are now a player in a game of Guandan. The game rules are as follows:

1. The game is played by four players in partnerships, sitting opposite each other.
2. The deck consists of two standard international decks with Jokers, totaling 108 cards.
3. The objective is to play higher combinations of cards to empty your hand before your opponents.
4. If your team completes the game first, you will advance in levels; the ultimate goal is to win on Level A.
5. Card ranks in increasing order are: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A.
6. There are four suits (Spades, Hearts, Diamonds, Clubs) and four Jokers (two red, two black).
7. Players take turns in counterclockwise order, starting from a player who plays any combination of cards.
8. Other players must play higher cards of the same type or a higher combination, or they must pass.
9. The game continues until three players have finished their cards.
10. Players are given titles based on the order they finish: Banker, Follower, Third, and Dweller.

The special cards and comparison are as follows:
1. Level Cards: The level number of the leading team determines the level cards. The level cards rank above aces but below jokers. For example, if the leading team is at level 6, then sixes are the level cards and rank above A.
2. Wild Cards: The two level cards in hearts are wild. During the round, they can be played as any card, except jokers, to form a combination with other cards. However, they only count as normal, non-wild cards when played as a single card. For example, when the level in the round is 7, the 7 of hearts can make a 4-bomb when combined with three 8s.
3. Normal Comparison: The normal comparison of the cards is from high to low in the order of red joker, black joker, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2. It applies when comparing with a single card, pair, triple, tube, plate, straight, bomb, and straight flush. Specially, full house compares the triple in the combination only.
4. Bomb Comparison: Bomb depends on its number of cards. The smallest is a 4-bomb of 2s and the largest is an 8-bomb of aces. However, a 5-bomb of 2s is larger than a 4-bomb of aces. A bomb ranks above: single card, pair, triple, tube, plate, full house, straight. A straight flush is regarded as a bomb that ranks above a 4 or 5-card bomb, except the joker bomb. A bomb with 6 or more cards ranks above a straight flush. Straight flushes rank according to their largest card regardless of suits. The joker bomb is the largest bomb in the game.

The representation of cards and card types is as follows:
1. **Cards**: Represented by a two-character string, such as 'S2' which means Spade 2. Detailed description below:
   - **Suits**: Spades, Hearts, Clubs, and Diamonds are represented by the characters S, H, C, and D respectively. Specifically, the suit for the small Joker is S, and for the big Joker, it is H.
   - **Ranks**: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K are represented by A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K respectively. That is, the rank 10 is represented by the character T. Specifically, the rank for the small Joker is represented by the character B, and for the big Joker, it is represented by the character R.
   For example, 'S2' represents Spade 2, 'HQ' represents Heart Q; 'SB' represents the small Joker, 'HR' represents the big Joker, 'PASS' indicates a pass.

2. **Card Types**: [Type, Rank, Cards]
   A card type is represented by a list of three fixed parts: Type, Rank, and Cards.
   - **Type**: The type of card combination, represented as a string with possible values of ['Single', 'Pair', 'Trips', 'ThreePair', 'ThreeWithTwo', 'TripsPair', 'Straight', 'Boom', 'PASS', 'tribute', 'back'].
   - **Rank**: The rank of the highest card or representative rank in the combination, with possible values of ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'B', 'R', 'PASS'].
   - **Cards**: The actual cards involved in the combination, represented as a list.
   Examples:
   - A single Diamond 5 is represented as: ['Single', '5', ['D5']].
   - A pair of 4s is represented as: ['Pair', '4', ['H4', 'C4']].
   - PASS: ['PASS', 'PASS', 'PASS'].

Your task is to make the best decision in each playing round. I will provide you with the following information:

1. Your position:
%s

2. Your current hand:
%s

3. Remaining cards of other players:
%s

4. Last action of other players:
%s

5. Last action of the teammate:
%s

6. Number of cards left for other players:
%s

7. Cards played by the down player:
%s

8. Cards played by the teammate:
%s

9. Cards played by the up player:
%s

10. Self rank:
%s

11. Opponent rank:
%s

12. Current rank:
%s

13. Legal actions:
%s

Please tell me your action in JSON format based on the provided information. The JSON should contain an "action" key with a value chose from legal actions.

Output format examples:
Playing a card: {"action": ["Single", "9", ["H9"]]}

Please provide the corresponding JSON action based on the given information.
'''
