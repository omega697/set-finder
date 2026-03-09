package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.card.SetCard

/**
 * Utility for interpreting test asset filenames.
 */
object SetCardTestUtils {

    /**
     * Parses a filename like "ONE_RED_EMPTY_DIAMOND_0.jpg" into a SetCard.
     * Supports both new standard and legacy "card_1_..." or "test_..." formats.
     */
    fun parseLabelFromFilename(filename: String): SetCard? {
        val cleanName = filename.removeSuffix(".jpg")
        val parts = cleanName.split("_")
        
        // Find where the attributes start (skip test_, card_, chip_)
        var offset = 0
        while (offset < parts.size && (parts[offset].lowercase() in listOf("test", "card", "chip"))) {
            offset++
        }
        
        if (parts.size - offset < 4) return null
        
        try {
            // Count
            val countStr = parts[offset]
            val count = if (countStr.length == 1) {
                when(countStr) { "1" -> SetCard.Count.ONE; "2" -> SetCard.Count.TWO; "3" -> SetCard.Count.THREE; else -> SetCard.Count.ONE }
            } else SetCard.Count.valueOf(countStr.uppercase())
            
            // Color
            val color = SetCard.Color.valueOf(parts[offset + 1].uppercase())
            
            // Pattern
            val patRaw = parts[offset + 2].lowercase()
            val pattern = when(patRaw) {
                "striped", "shaded" -> SetCard.Pattern.SHADED
                "open", "empty" -> SetCard.Pattern.EMPTY
                "solid" -> SetCard.Pattern.SOLID
                else -> SetCard.Pattern.valueOf(patRaw.uppercase())
            }
            
            // Shape
            val shape = SetCard.Shape.valueOf(parts[offset + 3].uppercase())
            
            return SetCard(shape, pattern, count, color)
        } catch (e: Exception) {
            android.util.Log.e("TestUtils", "Failed to parse filename: $filename", e)
            return null
        }
    }
    
    /**
     * Returns a string representation compatible with CardExpertModelInstrumentationTest.
     */
    fun formatLabel(card: SetCard): String {
        return "${card.count} ${card.color} ${card.pattern} ${card.shape}"
    }
}
