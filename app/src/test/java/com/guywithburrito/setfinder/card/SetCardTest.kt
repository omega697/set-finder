package com.guywithburrito.setfinder.card

import org.junit.Test
import com.google.common.truth.Truth.assertThat

/**
 * This test evaluates the game-rule logic for identifying valid sets. It verifies 
 * the 'isSet' rule implementation and the 'findSets' orchestration, ensuring 
 * that the app correctly identifies winning card combinations.
 */
class SetCardTest {

    @Test
    fun isSet_validSets_returnTrue() {
        val c1 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        
        assertThat(SetCard.isSet(c1, c2, c3)).isTrue()
        assertThat(Triple(c1, c2, c3).isSet()).isTrue()
    }

    @Test
    fun isSet_validMixedSets_returnTrue() {
        val card1 = SetCard(SetCard.Shape.SQUIGGLE, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.PURPLE)
        val card2 = SetCard(SetCard.Shape.SQUIGGLE, SetCard.Pattern.EMPTY, SetCard.Count.ONE, SetCard.Color.RED)
        val card3 = SetCard(SetCard.Shape.SQUIGGLE, SetCard.Pattern.SHADED, SetCard.Count.ONE, SetCard.Color.GREEN)

        assertThat(SetCard.isSet(card1, card2, card3)).isTrue()
    }

    @Test
    fun isSet_invalidSets_returnFalse() {
        val card1 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.PURPLE)
        val card2 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.EMPTY, SetCard.Count.ONE, SetCard.Color.RED)
        val card3 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SHADED, SetCard.Count.THREE, SetCard.Color.GREEN)

        assertThat(SetCard.isSet(card1, card2, card3)).isFalse()
        assertThat(Triple(card1, card2, card3).isSet()).isFalse()
    }

    @Test
    fun findSets_detectsCorrectSets() {
        val c1 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        val c4 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.EMPTY, SetCard.Count.ONE, SetCard.Color.PURPLE)
        
        val cards = listOf(c1, c2, c3, c4)
        val sets = SetCard.findSets(cards)
        
        assertThat(sets).hasSize(1)
        assertThat(sets[0]).containsExactly(c1, c2, c3)
    }
}
