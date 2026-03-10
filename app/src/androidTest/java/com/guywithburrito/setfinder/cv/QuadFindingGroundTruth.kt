package com.guywithburrito.setfinder.cv

import org.opencv.core.Point

/**
 * Manually identified ground truth for Quad Finding.
 * Coordinates are stored as percentages (0-100) of the image dimensions.
 */
object QuadFindingGroundTruth {
    data class GTCard(val pointsPercent: List<Point>)

    private val rawData: Map<String, List<List<Pair<Double, Double>>>> = mapOf(
        "card_1_purple_empty_oval.jpg" to listOf(
            listOf(25.0 to 34.0, 77.0 to 34.0, 77.0 to 75.0, 25.0 to 75.0)
        ),
        "card_1_red_shaded_diamond.jpg" to listOf(
            listOf(40.0 to 42.0, 72.0 to 46.0, 68.0 to 68.0, 36.0 to 63.0)
        ),
        "card_2_green_shaded_oval.jpg" to listOf(
            listOf(30.0 to 38.0, 72.0 to 38.0, 72.0 to 71.0, 30.0 to 71.0)
        ),
        "card_2_purple_solid_diamond.jpg" to listOf(
            listOf(34.0 to 32.0, 68.0 to 32.0, 68.0 to 76.0, 34.0 to 76.0)
        ),
        "card_3_green_solid_squiggle.jpg" to listOf(
            listOf(32.0 to 33.0, 74.0 to 32.0, 73.0 to 67.0, 30.0 to 69.0)
        ),
        "card_3_red_empty_squiggle.jpg" to listOf(
            listOf(22.0 to 28.0, 83.0 to 28.0, 83.0 to 75.0, 22.0 to 75.0)
        ),
        "card_3_red_solid_oval.jpg" to listOf(
            listOf(34.0 to 38.0, 70.0 to 38.0, 70.0 to 59.0, 34.0 to 59.0)
        ),
        "cards_3_no_set.jpg" to listOf(
            listOf(14.0 to 42.0, 25.0 to 41.0, 24.0 to 64.0, 13.0 to 65.0),
            listOf(34.0 to 35.0, 43.0 to 28.0, 50.0 to 51.0, 40.0 to 55.0),
            listOf(72.0 to 34.0, 82.0 to 32.0, 85.0 to 56.0, 74.0 to 58.0)
        ),
        "cards_8_1_set.jpg" to listOf(
            listOf(24.0 to 34.0, 32.0 to 34.0, 32.0 to 52.0, 24.0 to 52.0),
            listOf(40.0 to 31.0, 49.0 to 32.0, 47.0 to 48.0, 38.0 to 47.0),
            listOf(56.0 to 30.0, 65.0 to 31.0, 64.0 to 47.0, 55.0 to 46.0),
            listOf(71.0 to 28.0, 80.0 to 29.0, 79.0 to 45.0, 70.0 to 44.0),
            listOf(22.0 to 63.0, 31.0 to 63.0, 30.0 to 83.0, 21.0 to 83.0),
            listOf(40.0 to 59.0, 48.0 to 60.0, 47.0 to 79.0, 39.0 to 78.0),
            listOf(57.0 to 59.0, 65.0 to 60.0, 64.0 to 79.0, 56.0 to 78.0),
            listOf(74.0 to 57.0, 82.0 to 58.0, 81.0 to 78.0, 73.0 to 77.0)
        ),
        "cards_8_no_set.jpg" to listOf(
            listOf(8.0 to 24.0, 18.0 to 24.0, 17.0 to 46.0, 7.0 to 46.0),
            listOf(36.0 to 23.0, 46.0 to 23.0, 46.0 to 44.0, 36.0 to 44.0),
            listOf(58.0 to 20.0, 68.0 to 20.0, 68.0 to 41.0, 58.0 to 41.0),
            listOf(79.0 to 14.0, 89.0 to 14.0, 89.0 to 34.0, 79.0 to 34.0),
            listOf(8.0 to 63.0, 20.0 to 63.0, 18.0 to 88.0, 6.0 to 88.0),
            listOf(36.0 to 60.0, 47.0 to 60.0, 46.0 to 85.0, 35.0 to 85.0),
            listOf(56.0 to 60.0, 73.0 to 60.0, 73.0 to 74.0, 56.0 to 74.0),
            listOf(80.0 to 46.0, 91.0 to 46.0, 90.0 to 70.0, 79.0 to 70.0)
        ),
        "cards_9_2_sets.jpg" to listOf(
            listOf(16.0 to 23.0, 26.0 to 23.0, 25.0 to 31.0, 16.0 to 31.0),
            listOf(38.0 to 20.0, 50.0 to 20.0, 50.0 to 35.0, 38.0 to 35.0),
            listOf(62.0 to 18.0, 75.0 to 25.0, 68.0 to 35.0, 55.0 to 28.0),
            listOf(18.0 to 42.0, 27.0 to 42.0, 27.0 to 53.0, 17.0 to 53.0),
            listOf(36.0 to 38.0, 43.0 to 35.0, 49.0 to 45.0, 41.0 to 48.0),
            listOf(69.0 to 39.0, 78.0 to 40.0, 80.0 to 51.0, 70.0 to 51.0),
            listOf(15.0 to 62.0, 25.0 to 59.0, 28.0 to 74.0, 18.0 to 76.0),
            listOf(37.0 to 61.0, 47.0 to 61.0, 48.0 to 75.0, 38.0 to 76.0),
            listOf(63.0 to 69.0, 72.0 to 66.0, 79.0 to 78.0, 70.0 to 82.0)
        ),
        "cards_12_3_sets.jpg" to listOf(
            listOf(14.0 to 27.0, 28.0 to 24.0, 30.0 to 31.0, 16.0 to 33.0),
            listOf(39.0 to 25.0, 53.0 to 25.0, 53.0 to 31.0, 39.0 to 31.0),
            listOf(66.0 to 22.0, 78.0 to 23.0, 78.0 to 29.0, 65.0 to 29.0),
            listOf(15.0 to 39.0, 30.0 to 38.0, 31.0 to 45.0, 16.0 to 45.0),
            listOf(43.0 to 35.0, 58.0 to 32.0, 61.0 to 45.0, 46.0 to 48.0),
            listOf(66.0 to 39.0, 79.0 to 38.0, 79.0 to 43.0, 66.0 to 44.0),
            listOf(8.0 to 58.0, 21.0 to 53.0, 25.0 to 60.0, 12.0 to 65.0),
            listOf(44.0 to 59.0, 57.0 to 57.0, 58.0 to 64.0, 45.0 to 65.0),
            listOf(66.0 to 58.0, 80.0 to 53.0, 82.0 to 60.0, 69.0 to 65.0),
            listOf(12.0 to 75.0, 29.0 to 76.0, 28.0 to 83.0, 11.0 to 82.0),
            listOf(45.0 to 74.0, 60.0 to 72.0, 61.0 to 79.0, 46.0 to 81.0),
            listOf(73.0 to 71.0, 82.0 to 68.0, 84.0 to 77.0, 74.0 to 80.0)
        ),
        "cards_13_wide_shot.jpg" to listOf(
            listOf(18.0 to 44.0, 25.0 to 44.0, 24.0 to 53.0, 17.0 to 53.0),
            listOf(31.0 to 43.0, 38.0 to 43.0, 37.0 to 52.0, 30.0 to 52.0),
            listOf(43.0 to 42.0, 49.0 to 42.0, 48.0 to 51.0, 42.0 to 51.0),
            listOf(54.0 to 42.0, 61.0 to 41.0, 60.0 to 50.0, 53.0 to 51.0),
            listOf(64.0 to 41.0, 71.0 to 41.0, 70.0 to 50.0, 63.0 to 50.0),
            listOf(75.0 to 40.0, 81.0 to 40.0, 80.0 to 49.0, 74.0 to 49.0),
            listOf(10.0 to 60.0, 17.0 to 60.0, 16.0 to 71.0, 9.0 to 71.0),
            listOf(22.0 to 58.0, 30.0 to 58.0, 29.0 to 69.0, 21.0 to 69.0),
            listOf(36.0 to 57.0, 43.0 to 57.0, 42.0 to 68.0, 35.0 to 68.0),
            listOf(48.0 to 57.0, 54.0 to 57.0, 53.0 to 67.0, 47.0 to 67.0),
            listOf(59.0 to 56.0, 66.0 to 56.0, 65.0 to 67.0, 58.0 to 67.0),
            listOf(69.0 to 55.0, 76.0 to 55.0, 75.0 to 66.0, 68.0 to 66.0),
            listOf(83.0 to 55.0, 90.0 to 55.0, 89.0 to 66.0, 82.0 to 66.0)
        ),
        "desk_no_cards.jpg" to emptyList(),
        "scene_two_green_shaded_diamond.jpg" to listOf(
            listOf(52.0 to 46.0, 77.0 to 57.0, 73.0 to 79.0, 47.0 to 67.0)
        )
    )

    val scenes: Map<String, List<GTCard>> = rawData.mapValues { (_, cards) ->
        cards.map { points ->
            GTCard(points.map { Point(it.first, it.second) })
        }
    }
}
