using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Rendering;
using UnityEngine.Serialization;
using UnityEngine.Timeline;
using Unity.Collections.LowLevel.Unsafe;
using Unity.VisualScripting;
using Google.Protobuf.WellKnownTypes;

// 상속 받을 베이스 클래스를 Agent로 지정
public class CarAgent : Agent
{
    private Transform tr;   // 에이전트의 위치, 회전, 크기 결정
    private Rigidbody rb;   // 에이전트가 물리 제어로 동작 (중력의 영향을 받는 사실적 움직임)
    private RayPerceptionSensorComponent3D raySensorComponent;

    public float moveSpeed;
    public float turn;

    Vector3 startPosition;  // 시작 위치
    Vector3 startRotation;  // 시작 방향
    Vector3 currentPosition;    // 현재 위치
    private float dis_Traveled = 0; // 이동한 거리

    float dis_front = 0;
    float dis_left = 0;
    float dis_right = 0;
    float dis_leftfront = 0;
    float dis_rightfront = 0;
    float[] distances = new float[5];
    public override void Initialize()   // 초기화 메소드
    {
        tr = GetComponent<Transform>(); // 이 스크립트의 에이전트가 가진 Transform 컴포넌트를 tr에 저장
        rb = GetComponent<Rigidbody>();
        raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();

        startPosition = tr.position;
        startRotation = tr.eulerAngles;
    }

    public override void OnEpisodeBegin()   // 에피소프(학습)이 시작될 때마다 호출되는 메소드
    {
        // 속도와 각속도 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 위치 초기화 
        tr.position = startPosition;
        tr.eulerAngles = startRotation;
    }

    public override void CollectObservations(VectorSensor sensor) // 인공지능이 학습을 하기 위해 필요한 정보값을 넘겨주기 위해 사용한 함수 
    {
       

    }


    public override void OnActionReceived(ActionBuffers actions)    // 정책으로 전달받은 데이터를 기반으로 행동 실행
    {
        var action = actions.DiscreteActions[0];    // 불연속적 값
        currentPosition = transform.position;   // Action 적용 에이전트의 월드 공간 위치

        switch (action)
        {
            case 0: turn = 0.0f; moveSpeed = 5.5f; break;
            case 1: turn = 0.0f; moveSpeed = 9.0f; break;
            case 2: turn = 0.0f; moveSpeed = 12.5f; break;
            case 3: turn = -0.8f; moveSpeed = 5.5f; break;
            case 4: turn = -0.8f; moveSpeed = 9.0f; break;
            case 5: turn = -0.8f; moveSpeed = 12.5f; break;
            case 6: turn = 0.8f; moveSpeed = 5.5f; break;
            case 7: turn = 0.8f; moveSpeed = 9.0f; break;
            case 8: turn = 0.8f; moveSpeed = 12.5f; break;
            case 9: turn = -1.5f; moveSpeed = 5.5f; break;
            case 10: turn = -1.5f; moveSpeed = 9.0f; break;
            case 11: turn = -1.5f; moveSpeed = 12.5f; break;
            case 12: turn = 1.5f; moveSpeed = 5.5f; break;
            case 13: turn = 1.5f; moveSpeed = 9.0f; break;
            case 14: turn = 1.5f; moveSpeed = 12.5f; break;
            case 15: turn = 0.0f; moveSpeed *= 0.7f; break;
        }

        transform.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward); // Time.fixedDelatTime: 물리 고정 프레임 속도 업데이트가 수행되는 초 단위 간격
        transform.Rotate(new Vector3(0f, turn, 0f));


        // action 적용
        var input = raySensorComponent.GetRayPerceptionInput(); // ray sensor로 얻는 데이터를 input에 저장
        var output = RayPerceptionSensor.Perceive(input); // ray cast (광선 투사) 결과를 output에 저장 

        for (var rayIndex = 0; rayIndex < output.RayOutputs.Length; rayIndex++)
        {
            var extent = input.RayExtents(rayIndex); // 주어진 rayindex에 의한 시작/종료 포인트을 extent에 저장 
            Vector3 startPositionWorld = extent.StartPositionWorld; // rayindex의 시작 포인트를 해당 변수에 저장
            Vector3 endPositionWorld = extent.EndPositionWorld; // rayindex의 종료 포인트를 해당 변수에 저장

            var rayOutput = output.RayOutputs[rayIndex];

            float distance = 0f;

            if (rayOutput.HasHit && rayOutput.HitGameObject.CompareTag("wall"))
            {
                Vector3 hitPosition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);
                distance = Vector3.Distance(hitPosition, startPositionWorld);
            }
            else
            {
                distance = Vector3.Distance(endPositionWorld, startPositionWorld);
            }

            // 거리를 해당 인덱스에 저장
            if (rayIndex == 0)
            {
                distances[0] = distance;
            }
            else if (rayIndex == 18)
            {
                distances[1] = distance;
            }
            else if (rayIndex == 17)
            {
                distances[2] = distance;
                //Debug.Log("dis_right : "+ distance);
            }
            else if (rayIndex == 4)
            {
                distances[3] = distance;
            }
            else if (rayIndex == 3)
            {
                distances[4] = distance;
                //Debug.Log("dis_rightfront : "+ distance);
            }
        }
        dis_front = distances[0];
        dis_left = distances[1];
        dis_right = distances[2];
        dis_leftfront = distances[3];
        dis_rightfront = distances[4];

        dis_Traveled = Vector3.Distance(transform.position, currentPosition);   // 이동한 거리 계산: 에이전트의 action전의 공간 위치와 이동 후 현재 공간 위치 간의 거리 반환 
        // dis_Traveled *= 10;

        // 차량 scale: (0.5, 0.5, 0.5)
        

       if (dis_right > 0.789f && dis_right <= 1.435f) // 2차선
        {
            if (dis_rightfront >= 2.26f)
            {
                // 정상 주행
                SetReward(1.2f * dis_Traveled);
            }
            else if (dis_rightfront < 2.26f)
            {
                // 위험 주행
                SetReward(-0.4f * dis_Traveled);

            }
        }

        else if (dis_right > 2.815f && dis_right <= 3.665f) // 1차선 
        {
            SetReward(1.2f * dis_Traveled);

        }
        // 중앙선 침범
        else if (dis_right > 3.565f) // 중앙선
        {
            SetReward(-4.0f * dis_Traveled);

        }

        if (action == 15)
        {
            AddReward(-0.2f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        /*var actionOut = actionsOut.DiscreteActions[0];
        // 왼쪽
        if (Input.GetKey(KeyCode.LeftArrow)) actionOut = 9;

        // 직진
        if (Input.GetKey(KeyCode.UpArrow)) actionOut = 0;

        // 오른쪽
        if (Input.GetKey(KeyCode.RightArrow)) actionOut = 12;*/
    }

    private void OnCollisionEnter(Collision collision)  // 두 개의 물체가 서로 충돌한 순간에 호출 (물리적인 충돌에 반응, Rigidbody가 있는 물체끼리 충돌할 때)
    {
        // 연석에 충돌할 경우
        if (collision.collider.CompareTag("wall"))
        {
            SetReward(-10.0f);
            EndEpisode();
        }
        // 목표지점에 도달할 경우
        if (collision.collider.CompareTag("finalgoal"))
        {
            SetReward(15.0f);
            EndEpisode();
        }
       
        // 도로를 따라서 잘 주행하는 경우
        if (collision.collider.CompareTag("goal"))
        {
            SetReward(3.0f);
        }
    }
    private void OnTriggerEnter(Collider other)  // 한 개의 물체가 Collider Component를 가진 다른 물체의 Trigger Collider안으로 들어갔을 떄 호출 (Rigidbody가 없는 물체와의 충돌에도 사용) 
    {
        // 장애물(차량)에 충돌할 경우
        if (other.CompareTag("obstacle"))
        {
            SetReward(-10.0f);
            EndEpisode();
        }
    }
}