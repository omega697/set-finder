package com.guywithburrito.setfinder.ml

import com.guywithburrito.setfinder.card.SetCard
import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * This test evaluates the CardModelMapper, ensuring that raw ML output indices 
 * are correctly translated into domain-specific SetCard traits (Color, Shape, 
 * Count, Pattern). It also verifies the robustness of the argmax calculation 
 * and handling of invalid indices.
 */
class CardModelMapperTest {

    @Test
    fun v12_mapping_correctlyResolvesAttributes() {
        val mapper = CardModelMapper.V12
        
        // RED, OVAL, ONE, SOLID (Indices: 1, 1, 1, 1)
        val result = mapper.mapIndices(1, 1, 1, 1)
        assertThat(result).isEqualTo(SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED))
        
        // GREEN, DIAMOND, TWO, SHADED (Indices: 2, 2, 2, 2)
        val result2 = mapper.mapIndices(2, 2, 2, 2)
        assertThat(result2).isEqualTo(SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SHADED, SetCard.Count.TWO, SetCard.Color.GREEN))
        
        // PURPLE, SQUIGGLE, THREE, EMPTY (Indices: 3, 3, 3, 3)
        val result3 = mapper.mapIndices(3, 3, 3, 3)
        assertThat(result3).isEqualTo(SetCard(SetCard.Shape.SQUIGGLE, SetCard.Pattern.EMPTY, SetCard.Count.THREE, SetCard.Color.PURPLE))
    }

    @Test
    fun v12_mapping_returnsNullOnInvalidIndex() {
        val mapper = CardModelMapper.V12
        // Index 0 is Background/None
        assertThat(mapper.mapIndices(0, 1, 1, 1)).isNull()
        // Index 4 is out of bounds
        assertThat(mapper.mapIndices(1, 4, 1, 1)).isNull()
    }

    @Test
    fun argmax_returnsCorrectIndex() {
        val mapper = CardModelMapper.V12
        assertThat(mapper.argmax(floatArrayOf(0.1f, 0.8f, 0.1f))).isEqualTo(1)
        assertThat(mapper.argmax(floatArrayOf(0.9f, 0.05f))).isEqualTo(0)
        assertThat(mapper.argmax(floatArrayOf(0.1f, 0.1f, 0.95f))).isEqualTo(2)
    }
}
