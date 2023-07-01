using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;


public class LaneDetection : MonoBehaviour
{
    private Transform tr;
    private Rigidbody rb;
    private float moveSpeed = 10f;
    private float Turn = 0f;
    private Vector3 direction;

    public float dis_Finish; // 종료 거리
    private float dis_Traveled;
    Vector3 startPosition;
    Vector3 startRotation;

    int step = 0;
    float dis_left = 0;
    float dis_right = 0;

    public void Awake() // 에피소드가 시작되기 전에 모든 변수 초기화하는 메소드 (start보다 먼저 호출됨)
    {
        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
        dis_Finish = 55f;       // 여기서 변수 초기화를 시켜줘야 FixedUpdate() 적용됨

        // 물리엔진 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 위치 초기화
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        

    }

    private void FixedUpdate() // 물리 업데이트와 관련된 작업 수행 메소드 
    {
        var raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();    // 해당 컴포넌트를 raySensorComponent에 저장
        var input = raySensorComponent.GetRayPerceptionInput(); // ray sensor로 얻는 데이터를 input에 저장
        var output = RayPerceptionSensor.Perceive(input); // ray cast (광선 투사) 결과를 output에 저장 
        step++;
        //Debug.Log("step"+step);
        for (var rayIndex = 0; rayIndex < output.RayOutputs.Length; rayIndex++)
        {
            var extent = input.RayExtents(rayIndex); // 주어진 rayindex에 의한 시작/종료 포인트을 extent에 저장 
            Vector3 startPositionWorld = extent.StartPositionWorld; // rayindex의 시작 포인트를 해당 변수에 저장
            Vector3 endPositionWorld = extent.EndPositionWorld; // rayindex의 종료 포인트를 해당 변수에 저장

            var rayOutput = output.RayOutputs[rayIndex];

            if (rayOutput.HasHit) // ray가 부딪혔을 때 거리 계산
            {
                // Lerp()로 spw + (epw-spw)*ht 이용한 보간 (두 점 사이의 선을 따라 어느 정도 지점을 찾는데 사용)
                Vector3 hitposition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);    // hitFraction은 적중 개체까지의 정규화된 거리 
                //Debug.DrawLine(startPositionWorld, hitposition, Color.red);

                if (rayIndex == 1)
                {
                    dis_right = Vector3.Distance(hitposition, startPositionWorld); // Vector3.Distance는 인자간 거리를 반환
                    //Debug.Log("dis_right :  " + Vector3.Distance(hitposition, startPositionWorld));
                }
                else if (rayIndex == 2)
                {
                    dis_left = Vector3.Distance(hitposition, startPositionWorld);
                    //Debug.Log("dis_left : " + Vector3.Distance(hitposition, startPositionWorld));

                }
               
            }
            
        }

        float steerRatio = dis_left - dis_right; //  오른:2.985535 왼:2.185532
        //Debug.Log("steerRatio : " + steerRatio);

        if (steerRatio <= 3.58f && steerRatio >= 2.97f) // 1차선 도로를 유지하는
        {
            Turn = 0;
            moveSpeed = 10f;

        }
        else if (steerRatio < 2.97f && steerRatio > 0f) // 중앙선쪽으로 이동할 때
        {
            Turn = Mathf.Clamp(dis_left - dis_right + 0.05f, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio < 10.85f && steerRatio > 3.58f) // 연석쪽으로 이동할 때
        {
            Turn = Mathf.Clamp(dis_right - dis_left - 0.05f, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio < 11.19 && steerRatio >= 10.85f) //  중앙선 넘어갈떄
        {
            Turn = Mathf.Clamp(dis_left - dis_right , -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio >= 11.19) // 연석쪽에서 멀어져야할 때
        {
            Turn = Mathf.Clamp(dis_right - dis_left, -1.2f, 1.2f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio <= 0f) // 중앙선에서 멀어져야 할 때
        {
            Turn = Mathf.Clamp(dis_right - dis_left, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
      

        tr.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward); // Translate는 객체를 이동시키는 메소드
        tr.Rotate(new Vector3(0f, Turn, 0f));    // Rotate는 객체를 회전시키는 메소드 

        dis_Traveled = Vector3.Distance(tr.position, startPosition);
        //Debug.Log("dis_Traveled : " + dis_Traveled);

        if (dis_Traveled >= dis_Finish)
        {
            //Debug.Log("Finish");

            tr.position = startPosition;
            tr.eulerAngles = startRotation;
            moveSpeed = 10f;
            dis_Traveled = 0f;
            step = 0;
        }


    }

    /*private void OnTriggerEnter(Collider other) // 충돌 시 실행되는 메소드 
    {
        if (other.CompareTag("wall")) // 연석과 충돌 시
        {
            // 물리엔진 초기화
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            // 위치 초기화
            tr.position = startPosition;
            tr.eulerAngles = startRotation;

        }
    }*/

}
