LIBRARY IEEE;
USE ieee.numeric_std.ALL;
USE IEEE.MATH_REAL.ALL;
USE IEEE.STD_LOGIC_1164.ALL;
PACKAGE type_pkg IS
    TYPE a_t IS ARRAY(NATURAL RANGE <>) OF Real;
END PACKAGE type_pkg;
USE work.type_pkg.ALL;
LIBRARY IEEE;
USE ieee.numeric_std.ALL;
USE IEEE.MATH_REAL.ALL;
USE IEEE.STD_LOGIC_1164.ALL;
ENTITY neural_network IS
    GENERIC (
        n : NATURAL := 8
    );
    PORT (
        rst : IN INTEGER := 0;
        clk : IN STD_LOGIC;
        Gate : IN INTEGER := 1;
        epoch : IN INTEGER := 100;
        learning_rate : IN Real := 0.01);
END neural_network;

ARCHITECTURE Behavioral OF neural_network IS

    SIGNAL trainingDone : INTEGER := 0;
    SIGNAL initDone : INTEGER := 0;
    TYPE weight1_type IS ARRAY (0 TO 3, 0 TO 3) OF Real;
    TYPE weight2_type IS ARRAY (0 TO 2, 0 TO 3) OF Real;
    TYPE weight2t_type IS ARRAY (0 TO 3, 0 TO 2) OF Real;

    TYPE bias1_type IS ARRAY (0 TO 3) OF Real;
    TYPE bias2_type IS ARRAY (0 TO 2) OF Real;

    TYPE Z1 IS ARRAY (0 TO 3, 0 TO 134) OF Real;
    TYPE Z1_T IS ARRAY (0 TO 134, 0 TO 3) OF Real;
    TYPE Z2 IS ARRAY (0 TO 2, 0 TO 134) OF Real;
    TYPE Z2_T IS ARRAY (0 TO 134, 0 TO 2) OF Real;
    TYPE typ3_1 IS ARRAY (0 TO 4) OF Real;

    SIGNAL W1_main : weight1_type := ((-0.09876633, -0.35718596, 0.44722986, 0.29352748),
    (0.34627688, 0.38164043, 0.03911656, -0.42392313),
    (0.05500931, -0.3345927, 0.33755386, 0.47665125),
    (-0.21876252, -0.11775452, -0.4798053, -0.22671503));
    SIGNAL W2_main : weight1_type := ((0.44494522, -0.3614905, 0.08582211, -0.01805699),
    (0.4764886, -0.2312969, -0.13747156, 0.46826363),
    (0.11666375, 0.37380087, 0.03397125, -0.2076878),
    (-0.33720523, 0.1584909, -0.00739676, -0.42956752));
    SIGNAL W3_main : weight2_type := ((0.25442195, 0.32047808, 0.20619202, 0.13625813),
    (0.28977716, 0.3071915, -0.15820241, 0.21142566),
    (0.14794767, -0.4400105, 0.49743372, -0.17351103));

    SIGNAL b1_main : bias1_type := (0.24702251, -0.18307883, 0.39587706, -0.4715252);
    SIGNAL b2_main : bias1_type := (0.4114132, 0.43673313, -0.4447255, 0.32150942);
    SIGNAL b3_main : bias2_type := (-0.03898543, 0.30346006, 0.13150895);

    SIGNAL vEpoch : INTEGER := epoch;

    TYPE X_type IS ARRAY (0 TO 134, 0 TO 3) OF Real;
    TYPE X_test_type IS ARRAY (0 TO 14, 0 TO 3) OF Real;
    TYPE Y_type IS ARRAY (0 TO 134) OF Real;
    TYPE Y_test_type IS ARRAY (0 TO 14) OF Real;
    TYPE Yt_type IS ARRAY (0 TO 134, 0 TO 2) OF Real;

    SIGNAL X : X_type := (( 0.8 ,-0.6 , 0.5 , 0.4),
    (-1.6 ,-1.7 ,-1.4 ,-1.2),
    (-0.2 ,-0.4 , 0.3 , 0.1),
    ( 1.0  , 0.6 , 1.1 , 1.7),
    (-0.4 ,-1.3 , 0.1 , 0.1),
    ( 1.0  ,-0.1 , 0.7 , 0.7),
    ( 0.7 ,-0.4 , 0.3 , 0.1),
    ( 0.7 , 0.3 , 0.9 , 1.4),
    (-0.3 ,-0.8 , 0.3 , 0.1),
    ( 0.6 ,-1.7 , 0.4 , 0.1),
    (-1.4 , 0.3 ,-1.2 ,-1.3),
    (-1.4 , 0.3 ,-1.4 ,-1.3),
    ( 0.3 ,-0.1 , 0.6 , 0.8),
    (-0.9 , 1.7 ,-1.1 ,-1.1),
    ( 1.6 , 0.3 , 1.3 , 0.8),
    ( 0.4 ,-0.6 , 0.6 , 0.8),
    (-0.9 , 0.6 ,-1.2 ,-0.9),
    (-0.3 ,-0.4 ,-0.1 , 0.1),
    ( 2.2 , 1.7 , 1.7 , 1.3),
    (-0.5 ,-0.1 , 0.4 , 0.4),
    (-0.1 , 2.2 ,-1.5 ,-1.3),
    ( 0.7 ,-0.6 , 1.0  , 1.3),
    ( 0.3 ,-0.1 , 0.5 , 0.3),
    (-0.4 ,-1.7 , 0.1 , 0.1),
    ( 1.3 , 0.1 , 0.8 , 1.4),
    ( 0.2 ,-0.4 , 0.4 , 0.4),
    ( 0.4 ,-0.4 , 0.3 , 0.1),
    ( 1.2 , 0.3 , 1.2 , 1.4),
    (-1.0  ,-1.7 ,-0.3 ,-0.3),
    (-1.3 , 0.8 ,-1.2 ,-1.3),
    ( 0.3 ,-1.1 , 1.0  , 0.3),
    (-0.8 , 0.8 ,-1.3 ,-1.3),
    ( 1.0  , 0.1 , 0.4 , 0.3),
    (-0.9 , 1.0  ,-1.3 ,-1.3),
    (-0.8 , 1.0  ,-1.3 ,-1.3),
    (-0.4 , 1.0  ,-1.4 ,-1.3),
    (-1.1 ,-1.3 , 0.4 , 0.7),
    ( 0.4 , 0.8 , 0.9 , 1.4),
    ( 2.5 , 1.7 , 1.5 , 1.1),
    (-0.5 , 0.8 ,-1.2 ,-1.3),
    (1.0 , 0.1 , 1.0 , 1.6),
    ( 0.6 , 0.8 , 1.0  , 1.6),
    ( 2.1 ,-0.1 , 1.6 , 1.2),
    ( 0.2 ,-2.0  , 0.7 , 0.4),
    (-1.9 ,-0.1 ,-1.5 ,-1.4),
    (-1.3 , 0.1 ,-1.2 ,-1.3),
    (-0.1 ,-0.8 , 0.8 , 0.9),
    (-0.2 ,-0.6 , 0.2 , 0.1),
    (-0.8 ,-0.8 , 0.1 , 0.3),
    ( 2.2 ,-0.1 , 1.3 , 1.4),
    ( 0.6 ,-0.8 , 0.6 , 0.8),
    ( 0.3 ,-0.6 , 0.1 , 0.1),
    (-0.9 , 1.5 ,-1.3 ,-1.1),
    (-0.2 ,-0.6 , 0.4 , 0.1),
    (-0.5 , 1.9 ,-1.4 ,-1.1),
    (-0.1 ,-0.6 , 0.8 , 1.6),
    (-0.1 ,-1.1 , 0.1 , 0.0 ),
    (-0.9 ,-1.3 ,-0.4 ,-0.1),
    (-0.2 , 1.7 ,-1.2 ,-1.2),
    (-0.4 ,-1.5 ,-0.0  ,-0.3),
    ( 0.1 ,-0.1 , 0.3 , 0.4),
    ( 1.0  , 0.1 , 0.5 , 0.4),
    ( 2.2 ,-0.6 , 1.7 , 1.1),
    (-1.1 ,-0.1 ,-1.3 ,-1.3),
    ( 0.6 , 0.6 , 0.5 , 0.5),
    (-0.9 , 1.7 ,-1.3 ,-1.2),
    (-0.1 ,-0.8 , 0.1 , 0.0 ),
    ( 1.3 , 0.3 , 1.1 , 1.4),
    (-1.1 , 0.1 ,-1.3 ,-1.4),
    (-1.0  ,-2.4 ,-0.1 ,-0.3),
    (-1.1 , 0.1 ,-1.3 ,-1.3),
    (-1.0  , 1.0  ,-1.4 ,-1.2),
    (-0.5 , 1.5 ,-1.3 ,-1.3),
    ( 1.2 ,-0.1 , 1.0  , 1.2),
    ( 0.6 , 0.6 , 1.3 , 1.7),
    (-1.0  , 0.8 ,-1.2 ,-1.1),
    (-0.8 , 2.4 ,-1.3 ,-1.4),
    ( 0.6 ,-1.3 , 0.7 , 0.9),
    ( 2.2 ,-1.1 , 1.8 , 1.4),
    ( 0.2 ,-2.0  , 0.1 ,-0.3),
    ( 1.6 , 1.2 , 1.3 , 1.7),
    (-1.1 ,-1.5 ,-0.3 ,-0.3),
    ( 0.3 ,-0.6 , 0.5 , 0.0 ),
    ( 0.8 ,-0.1 , 0.8 , 1.1),
    (-0.9 , 1.0  ,-1.3 ,-1.2),
    ( 1.9 ,-0.6 , 1.3 , 0.9),
    ( 0.1 ,-0.1 , 0.8 , 0.8),
    ( 0.7 ,-0.8 , 0.9 , 0.9),
    (-1.7 ,-0.4 ,-1.3 ,-1.3),
    (-0.3 ,-0.6 , 0.6 , 1.1),
    (-0.9 , 0.8 ,-1.3 ,-1.3),
    (-1.0  , 0.6 ,-1.3 ,-1.3),
    (-0.2 ,-1.1 ,-0.1 ,-0.3),
    ( 1.3 , 0.1 , 0.9 , 1.2),
    (-0.4 ,-1.1 , 0.4 , 0.0 ),
    ( 0.6 ,-0.6 , 0.8 , 0.4),
    ( 1.4 , 0.3 , 0.5 , 0.3),
    ( 1.2 ,-0.6 , 0.6 , 0.3),
    ( 0.8 , 0.3 , 0.8 , 1.1),
    ( 1.3 , 0.1 , 0.6 , 0.4),
    (-0.4 ,-1.5 , 0.0  ,-0.1),
    (-0.3 ,-0.1 , 0.4 , 0.4),
    ( 0.2 ,-0.1 , 0.6 , 0.8),
    ( 0.2 ,-0.8 , 0.8 , 0.5),
    (-0.1 ,-0.8 , 0.8 , 0.9),
    ( 0.1 , 0.3 , 0.6 , 0.8),
    (-0.9 , 1.7 ,-1.2 ,-1.3),
    ( 0.4 ,-2.0  , 0.4 , 0.4),
    (-0.3 ,-0.1 , 0.2 , 0.1),
    (-0.3 ,-1.3 , 0.1 ,-0.1),
    (-0.1 ,-0.8 , 0.2 ,-0.3),
    ( 0.7 , 0.1 , 1.0  , 0.8),
    ( 0.9 ,-0.4 , 0.5 , 0.1),
    (-0.4 , 2.6 ,-1.3 ,-1.3),
    (-1.7 ,-0.1 ,-1.4 ,-1.3),
    ( 0.6 ,-0.4 , 1.0  , 0.8),
    ( 0.7 , 0.3 , 0.4 , 0.4),
    ( 1.8 ,-0.4 , 1.4 , 0.8),
    (-1.0  , 1.2 ,-1.3 ,-1.3),
    ( 1.5 ,-0.1 , 1.2 , 1.2),
    (-1.1 , 1.2 ,-1.3 ,-1.4),
    ( 0.8 ,-0.1 , 1.0  , 0.8),
    ( 1.0  , 0.6 , 1.1 , 1.2),
    (-0.5 , 0.8 ,-1.3 ,-1.1),
    ( 0.3 ,-0.4 , 0.5 , 0.3),
    (-1.5 , 0.3 ,-1.3 ,-1.3),
    ( 1.0  ,-0.1 , 0.8 , 1.4),
    (-0.2 ,-0.1 , 0.3 , 0.0 ),
    (-1.0  , 0.8 ,-1.3 ,-1.3),
    ( 0.8 ,-0.1 , 1.2 , 1.3),
    ( 0.9 ,-0.1 , 0.4 , 0.3),
    (-1.0  , 1.0  ,-1.2 ,-0.8),
    (-1.5 , 1.2 ,-1.6 ,-1.3),
    ( 0.6 ,-1.3 , 0.6 , 0.4),
    (-0.2 , 3.1 ,-1.3 ,-1.1));

    SIGNAL X_test_reshape : X_test_type;

    SIGNAL Y : Y_type := (1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0,
    2.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0,
    1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0,
    1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0,
    0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0);

    SIGNAL X_test : X_test_type := 
    ((-1.5 , 0.8 ,-1.3 ,-1.2),
     (-1.5 , 0.1 ,-1.3 ,-1.3),
     (-0.2 ,-1.3 , 0.7 , 1.1),
     (-1.3 ,-0.1 ,-1.3 ,-1.4),
     (-1.3 , 0.8 ,-1.1 ,-1.3),
     ( 1.6 ,-0.1 , 1.2 , 0.5),
     (-1.0  ,-0.1 ,-1.2 ,-1.3),
     ( 1.0  ,-1.3 , 1.2 , 0.8),
     ( 0.7 ,-0.6 , 1.0  , 1.2),
     (-1.3 ,-0.1 ,-1.3 ,-1.2),
     (-0.7 , 1.5 ,-1.3 ,-1.3),
     (-1.7 , 0.3 ,-1.4 ,-1.3),
     (-1.0  ,0.3 ,-1.5 ,-1.3),
     (-0.5 , 1.9 ,-1.2 ,-1.1),
     ( 0.2 , 0.8 , 0.4 , 0.5));

    SIGNAL Y_test : Y_test_type := (0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    SIGNAL mcorrect : INTEGER;
    -- relu function implementation
    FUNCTION relu(x : real) RETURN real IS
    BEGIN
        IF x <= 0.0 THEN
            RETURN 0.0;
        ELSE
            RETURN x;
        END IF;
    END relu;
    FUNCTION relud(x : real) RETURN real IS
    BEGIN
        IF x <= 0.0 THEN
            RETURN 0.0;
        ELSE
            RETURN 1.0;
        END IF;
    END relud;

BEGIN

    --Forward Propagation
    PROCESS (rst, clk)
        VARIABLE W1 : weight1_type := W1_main;
        VARIABLE W2 : weight1_type := W2_main;
        VARIABLE W3 : weight2t_type ;
        VARIABLE b1 : bias1_type := b1_main;
        VARIABLE b2 : bias1_type := b2_main;
        VARIABLE b3 : bias2_type := b3_main;
        VARIABLE Z1, Z1F : bias1_type;
        VARIABLE h1, h1F : bias1_type;
        VARIABLE Z2, Z2F : bias1_type;
        VARIABLE h2, h2F : bias1_type;
        VARIABLE Z3, Z3F : bias2_type;
        VARIABLE h3, h3F : bias2_type;
        VARIABLE y_temp : Yt_type;
        VARIABLE output_error : bias2_type;
        VARIABLE output_delta : bias2_type;
        VARIABLE h2e : bias1_type;
        VARIABLE hidden_delta_2, hidden_delta_1 : bias1_type;
        VARIABLE output_bias_delta, hidden_bias_delta_2, hidden_bias_delta_1 : Real := 0.0;
        VARIABLE h1e : bias1_type;
        VARIABLE test_out : Y_type;
        VARIABLE correct : INTEGER;
        VARIABLE percision : Real;
    BEGIN
        IF rst = 1 THEN
            trainingDone <= 0;
            initDone <= 0;
            vEpoch <= epoch;
        ELSE
            IF initDone = 0 THEN
                initDone <= 1;
                FOR i IN 0 TO 3 LOOP
                    FOR j IN 0 TO 3 LOOP
                        W1(i, j) := W1_main(j, i);
                        W2(i, j) := W2_main(j, i);
                    END LOOP;
                END LOOP;
                FOR i IN 0 TO 3 LOOP
                    FOR j IN 0 TO 2 LOOP
                        W3(i, j) := W3_main(j, i);
                    END LOOP;
                END LOOP;
                FOR i IN 0 TO 134 LOOP

                    IF Y(i) = 0.0 THEN
                        y_temp(i, 0) := 1.0;
                        y_temp(i, 1) := 0.0;
                        y_temp(i, 2) := 0.0;
                    ELSIF Y(i) = 1.0 THEN
                        y_temp(i, 1) := 1.0;
                        y_temp(i, 0) := 0.0;
                        y_temp(i, 2) := 0.0;
                    ELSIF Y(i) = 2.0 THEN
                        y_temp(i, 2) := 1.0;
                        y_temp(i, 0) := 0.0;
                        y_temp(i, 1) := 0.0;
                    END IF;

                END LOOP;
            END IF;
            IF trainingDone = 0 THEN
                IF vEpoch > 0 THEN
                    -- Forward Propagattion
                    -- First layer - 4 input to 4 neurons for 135 sample
                    FOR p IN 0 TO 134 LOOP
                        output_bias_delta := 0.0;
                        hidden_bias_delta_2 := 0.0;
                        hidden_bias_delta_1 := 0.0;
                        h1e := (OTHERS => 0.0);
                        --Forward-firstlayer
                        FOR i IN 0 TO 3 LOOP
                            Z1(i) := W1(0, i) * X(p, 0) + W1(1, i) * X(p, 1) + W1(2, i) * X(p, 2) + W1(3, i) * X(p, 3) + b1(i);
                            h1(i) := relu(Z1(i));
                        END LOOP;
                        --Forward-secondlayer
                        FOR i IN 0 TO 3 LOOP
                            Z2(i) := W2(0, i) * h1(0) + W2(1, i) * h1(1) + W2(2, i) * h1(2) + W1(3, i) * h1(3) + b2(i);
                            h2(i) := relu(Z2(i));
                        END LOOP;
                        --Forward-thirdlayer
                        FOR i IN 0 TO 2 LOOP
                            Z3(i) := W3(0, i) * h2(0) + W3(1, i) * h2(1) + W3(2, i) * h2(2) + W3(3, i) * h2(3) + b3(i);
                            h3(i) := relu(Z3(i));
                            --error and gradinat output
                            output_error(i) := h3(i) - y_temp(p, i);
                            output_delta(i) := h3(i) * output_error(i);
                            output_bias_delta := output_bias_delta + output_error(i);
                        END LOOP;
                        -- Backward-out
                        --gardiant second layer
                        FOR i IN 0 TO 3 LOOP
                            for j in 0 to 2 loop
				h2e(i):= 0.0;
                                h2e(i) := h2e(i) + (output_error(j) * W2(0, i) * relud(h2(i)) + output_error(j) * W2(1, i) * relud(h2(i)) + output_error(j) * W2(2, i) * relud(h2(i)) + output_error(j) * W2(3, i) * relud(h2(i)));
                            end loop;
                            hidden_delta_2(i) := h1(i) * h2e(i);
                            hidden_bias_delta_2 := h2e(i) + hidden_bias_delta_2;
                        END LOOP;
                        -- gradiant first layer
                        -- hidden_error_1[i] = sum(hidden_error_2[m, j] * hidden_layer_2[m, j] * (hidden_layer_2[m, j] > 0) * hidden_layer_1[m, i])
                        FOR i IN 0 TO 3 LOOP
                            for j in 0 to 3 loop
                                h1e(i) := h1e(j)+ (h1(i) * (h2e(j) * h2(i) * relud(h2(i))));
                            end loop;                          
                            hidden_delta_1(i) := X(p, i) * h1e(i);
                            hidden_bias_delta_1 := h1e(i) + hidden_bias_delta_1;
                        END LOOP;

                        -- update weight
                        --first layer
                        FOR i IN 0 TO 3 LOOP
                            FOR j IN 0 TO 3 LOOP
                                W1(i, j) := W1(i, j) - (learning_rate * hidden_delta_1(j));
                                b1(i) := b1(i) - (learning_rate * hidden_bias_delta_1);
                            END LOOP;
                        END LOOP;
                        --first layer
                        FOR i IN 0 TO 3 LOOP
                            FOR j IN 0 TO 3 LOOP
                                W2(i, j) := W2(i, j) - (learning_rate * hidden_delta_2(j));
                                b2(i) := b2(i) - (learning_rate * hidden_bias_delta_2);

                            END LOOP;
                            FOR i IN 0 TO 2 LOOP
                                FOR j IN 0 TO 3 LOOP
                                    W3(j, i) := W3(j, i) - (learning_rate * output_delta(i));
                                    b3(i) := b3(i) - (learning_rate * output_bias_delta);

                                END LOOP;
                            END LOOP;
                        END LOOP;
                    END LOOP;
                    vEpoch <= vEpoch - 1;
                END IF;
                IF vEpoch <= 0 THEN
                    trainingDone <= 1;
                    -- FOR i IN 0 TO 3 LOOP
                    --     FOR j IN 0 TO 3 LOOP
                    --         W1_main(i, j) <= W1(j, i);
                    --         W2_main(i, j) <= W2(j, i);
                    --     END LOOP;
                    -- END LOOP;
                    -- FOR i IN 0 TO 3 LOOP
                    --     FOR j IN 0 TO 2 LOOP
                    --         W3_main(i, j) <= W3(j, i);
                    --     END LOOP;
                    -- END LOOP;
                    -- b1_main <= b1;
                    -- b2_main <= b2;
                    -- b3_main <= b3;
                END IF;
            END IF;
            IF trainingDone = 0 THEN
                --Forward-firstlayer
                correct := 0;
                FOR p IN 0 TO 14 LOOP
                    FOR i IN 0 TO 3 LOOP
                        Z1F(i) := W1(0, i) * X(p, 0) + W1(1, i) * X_Test(p, 1) + W1(2, i) * X(p, 2) + W1(3, i) * X(p, 3) + b1(i);
                        h1F(i) := relu(Z1F(i));
                    END LOOP;
                    --Forward-secondlayer
                    FOR i IN 0 TO 3 LOOP
                        Z2F(i) := W2(0, i) * h1F(0) + W2(1, i) * h1F(1) + W2(2, i) * h1F(2) + W1(3, i) * h1F(3) + b2(i);
                        h2F(i) := relu(Z2F(i));
                    END LOOP;
                    --Forward-thirdlayer
                    FOR i IN 0 TO 2 LOOP
                        Z3F(i) := W3(0, i) * h2F(0) + W3(1, i) * h2F(1) + W3(2, i) * h2F(2) + W3(3, i) * h2F(3) + b3(i);
                        h3F(i) := relu(Z3F(i));
                    END LOOP;
                    IF h3F(0) >= h3F(1) AND h3F(0) >= h3F(2) THEN
                        test_out(p) := 0.0;
                    ELSIF h3F(1) >= h3F(0) AND h3F(1) >= h3F(0) THEN
                        test_out(p) := 1.0;
                    ELSE
                        test_out(p) := 2.0;
                    END IF;
                    IF Y_test(p) = test_out(p) THEN
                        correct := correct + 1;
                    END IF;
                END LOOP;
                -- percision := correct/15.0;
            END IF;
        END IF;
        mcorrect <= correct;
    END PROCESS;
END Behavioral;



