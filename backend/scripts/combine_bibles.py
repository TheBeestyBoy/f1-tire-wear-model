#!/usr/bin/env python3
"""
Script to combine multiple Bible versions for training.
"""

import os

def combine_bible_versions():
    """Combine multiple Bible text files into one training file."""

    data_dir = "data"
    output_path = "data/training_text.txt"

    # List of Bible files to combine (add your files here)
    bible_files = [
        "data/kjv.txt",
        "data/net.txt",
        "data/tyndale.txt",
        "data/web.txt",
        "data/geneva.txt",
        "data/coverdale.txt",
        "data/bishops.txt",
        "data/asv.txt",
    ]

    # List of ALL classic book files (137 books)
    # Note: Files like as_you_like_it.txt, much_ado.txt, book_mormon.txt, etc.
    # are in download script but may not have downloaded successfully.
    # This script will automatically skip any missing files.
    classic_books = [
        # Shakespeare
        "data/hamlet.txt", "data/macbeth.txt", "data/romeo_juliet.txt", "data/julius_caesar.txt",
        "data/othello.txt", "data/merchant_venice.txt", "data/midsummer_nights_dream.txt",
        "data/tempest.txt", "data/twelfth_night.txt", "data/richard_iii.txt", "data/king_lear.txt",
        "data/henry_iv_part1.txt", "data/henry_v.txt", "data/henry_vi_part1.txt", "data/henry_viii.txt",
        "data/taming_of_shrew.txt",

        # Jane Austen
        "data/pride_prejudice.txt", "data/emma.txt", "data/persuasion.txt", "data/northanger_abbey.txt",
        "data/mansfield_park.txt", "data/lady_susan.txt", "data/sense_sensibility.txt",

        # Charles Dickens
        "data/oliver_twist.txt", "data/tale_two_cities.txt", "data/great_expectations.txt",
        "data/david_copperfield.txt", "data/christmas_carol.txt", "data/bleak_house.txt",
        "data/little_dorrit.txt", "data/our_mutual_friend.txt", "data/hard_times.txt",

        # Mark Twain
        "data/huckleberry_finn.txt", "data/tom_sawyer.txt", "data/prince_and_pauper.txt",
        "data/yankee_court.txt", "data/innocents_abroad.txt", "data/puddnhead_wilson.txt",

        # H.G. Wells
        "data/time_machine.txt", "data/war_worlds.txt", "data/invisible_man.txt",
        "data/island_dr_moreau.txt",

        # Jules Verne
        "data/twenty_thousand_leagues.txt", "data/around_world_80_days.txt",
        "data/journey_center_earth.txt", "data/mysterious_island.txt", "data/from_earth_moon.txt",

        # Russian Classics
        "data/crime_punishment.txt", "data/anna_karenina.txt", "data/war_peace.txt",
        "data/brothers_karamazov.txt", "data/idiot.txt", "data/demons.txt",
        "data/notes_underground.txt", "data/war_peace_tolstoy.txt", "data/resurrection.txt",

        # Edgar Allan Poe
        "data/raven.txt", "data/cask_amontillado.txt", "data/masque_red_death.txt",

        # Oscar Wilde
        "data/picture_dorian_gray.txt", "data/importance_earnest.txt", "data/happy_prince.txt",
        "data/canterville_ghost.txt",

        # Arthur Conan Doyle
        "data/sherlock_holmes.txt", "data/sherlock_study.txt", "data/sign_of_four.txt",
        "data/hound_baskervilles.txt", "data/valley_of_fear.txt", "data/memoirs_sherlock.txt",

        # Greek Classics
        "data/odyssey.txt", "data/iliad.txt", "data/aeneid.txt", "data/oedipus_king.txt",
        "data/antigone.txt", "data/agamemnon.txt", "data/medea.txt",

        # Epic Poetry
        "data/paradise_lost.txt", "data/paradise_regained.txt", "data/divine_comedy.txt",
        "data/leaves_of_grass.txt",

        # Philosophy
        "data/republic.txt", "data/republic_plato.txt", "data/apology_plato.txt", "data/phaedo.txt",
        "data/meditations_marcus.txt", "data/meditations_descartes.txt", "data/prince.txt",
        "data/prince_machiavelli.txt", "data/critique_pure_reason.txt", "data/tractatus.txt",
        "data/leviathan.txt", "data/thus_spoke_zarathustra.txt",

        # Religious/Spiritual
        "data/koran.txt", "data/summa_theologica.txt", "data/imitation_christ.txt",
        "data/pilgrims_progress.txt",

        # Gothic/Horror
        "data/dracula.txt", "data/dracula_full.txt", "data/frankenstein.txt",
        "data/jekyll_and_hyde.txt", "data/legend_sleepy_hollow.txt",

        # Victorian Literature
        "data/jane_eyre.txt", "data/wuthering_heights.txt", "data/middlemarch.txt",
        "data/middlemarch_eliot.txt", "data/scarlet_letter.txt", "data/house_seven_gables.txt",
        "data/villette.txt", "data/tess_durbervilles.txt",

        # Adventure
        "data/moby_dick.txt", "data/treasure_island.txt", "data/kidnapped.txt",
        "data/call_of_wild.txt", "data/last_mohicans.txt", "data/ivanhoe.txt",
        "data/scarlet_pimpernel.txt", "data/three_musketeers.txt", "data/man_iron_mask.txt",
        "data/count_monte_cristo.txt",

        # Other Classics
        "data/alice_wonderland.txt", "data/les_miserables.txt", "data/don_quixote.txt",
        "data/gullivers_travels.txt", "data/little_women.txt", "data/secret_garden.txt",
        "data/peter_pan.txt", "data/little_prince.txt", "data/heart_of_darkness.txt",
        "data/ulysses.txt", "data/utopia.txt", "data/walden.txt", "data/grimms_fairy_tales.txt",
        "data/yellow_wallpaper.txt", "data/tale_of_genji.txt", "data/great_gatsby.txt",

        # Science/Nature
        "data/origin_species.txt", "data/common_sense.txt", "data/essays_bacon.txt",
        "data/metamorphoses.txt",
    ]

    # Combine all files
    all_files = bible_files + classic_books

    combined_text = []
    total_chars = 0

    print("Combining Bible versions and classic books...")

    for filepath in all_files:
        if os.path.exists(filepath):
            print(f"  Reading {filepath}...")

            # Try UTF-8 first, then fall back to latin-1 or other encodings
            text = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        text = f.read()
                    break  # Success, stop trying encodings
                except (UnicodeDecodeError, UnicodeError):
                    continue  # Try next encoding

            if text is None:
                print(f"  [WARNING] Could not read {filepath} with any encoding, skipping...")
                continue

            combined_text.append(text)
            total_chars += len(text)
            print(f"    Added {len(text):,} characters")
        else:
            print(f"  [WARNING] File not found: {filepath}")

    if not combined_text:
        print("[ERROR] No files found!")
        return False

    # Combine with separator
    final_text = "\n\n=== NEW TEXT ===\n\n".join(combined_text)

    # Save combined version
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"\n[OK] Combined {len(all_files)} texts ({len(bible_files)} Bibles + {len(classic_books)} classic books)")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {len(final_text.split()):,}")
    print(f"Output: {output_path}")

    return True

if __name__ == "__main__":
    # Change to backend directory if needed
    if os.path.basename(os.getcwd()) != "backend":
        if os.path.exists("backend"):
            os.chdir("backend")

    combine_bible_versions()
