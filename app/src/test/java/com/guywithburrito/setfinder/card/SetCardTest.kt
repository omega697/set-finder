package com.guywithburrito.setfinder.card

import org.junit.Test
import com.google.common.truth.Truth.assertThat

class SetCardTest {
    @Test
    fun calculatesSets_isSet() {
        val card1 = SetCard(
            SetCard.Shape.SQUIGGLE,
            SetCard.Pattern.SOLID,
            SetCard.Count.ONE,
            SetCard.Color.PURPLE
        )
        val card2 = SetCard(
            SetCard.Shape.SQUIGGLE,
            SetCard.Pattern.EMPTY,
            SetCard.Count.ONE,
            SetCard.Color.RED
        )
        val card3 = SetCard(
            SetCard.Shape.SQUIGGLE,
            SetCard.Pattern.SHADED,
            SetCard.Count.ONE,
            SetCard.Color.GREEN
        )

        assertThat(SetCard.isSet(card1, card2, card3)).isTrue()
        assertThat(Triple(card1, card2, card3).isSet()).isTrue()
    }

    @Test
    fun calculatesSets_isNotSet() {
        val card1 = SetCard(
            SetCard.Shape.OVAL,
            SetCard.Pattern.SOLID,
            SetCard.Count.TWO,
            SetCard.Color.PURPLE
        )
        val card2 = SetCard(
            SetCard.Shape.DIAMOND,
            SetCard.Pattern.EMPTY,
            SetCard.Count.ONE,
            SetCard.Color.RED
        )
        val card3 = SetCard(
            SetCard.Shape.OVAL,
            SetCard.Pattern.SHADED,
            SetCard.Count.THREE,
            SetCard.Color.GREEN
        )

        assertThat(SetCard.isSet(card1, card2, card3)).isFalse()
        assertThat(Triple(card1, card2, card3).isSet()).isFalse()
    }
}